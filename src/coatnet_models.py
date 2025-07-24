import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from typing import Any, Tuple
from timm.models.layers import DropPath, create_conv2d
from timm.layers.std_conv import StdConv2d

class CoAtNetSideViTClassifier_1(nn.Module):
    """
    Enhanced CoAtNet + Side-ViT ensemble with advanced regularization to reduce overfitting:
      - DropPath (stochastic depth) in backbone
      - Feature dropout
      - Weight-standardized convolutions (StdConv2d)
      - Squeeze-and-Excitation adapters
      - LayerNorm & strong dropout in MLP head
    """
    def __init__(
        self,
        side_vit1,
        side_vit2,
        side_vit_cnn,
        cfg: Any,
        pretrained: bool = True,
    ):
        super().__init__()
        # --- Backbone: CoAtNet with stochastic depth & dropout ---
        self.backbone = timm.create_model(
            'coatnet_0_rw_224',
            pretrained=pretrained,
            features_only=True,
            drop_rate=0.1,
            drop_path_rate=0.2
        )
        # Freeze early layers, fine-tune later ones
        for name, param in self.backbone.named_parameters():
            param.requires_grad = True if 'layer3' in name or 'layer4' in name else False

        # Channel dims for CoAtNet stages
        c2, c3, c4 = 192, 384, 768
                # Determine input channels (e.g. 3 for RGB, 1 for grayscale)
        in_ch = getattr(cfg.dataset, 'image_channel_num', None)
        if not isinstance(in_ch, int) or in_ch < 1:
            # Fallback to grayscale if misconfigured
            in_ch = 1  # assume single-channel input
            print(f"Warning: invalid image_channel_num in config, fallback to in_ch={in_ch}")



## -----------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 bias=False, dropblock_prob=0.1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # Spatial regularization via Dropout or DropBlock
        self.dropblock = nn.Dropout2d(p=dropblock_prob)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.dropblock(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=24, dropout_p=0.1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # Apply stochastic depth to SE output for regularization
        self.drop_path = DropPath(drop_prob=0.1)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return self.drop_path(out)

class LightweightFPNFusion(nn.Module):
    def __init__(self, c2_dim, c3_dim, fusion_dim, out_dim,
                 droppath_prob=0.1, dropblock_prob=0.1):
        super().__init__()
        self.top_down_proj = nn.Conv2d(c3_dim, fusion_dim, kernel_size=1, bias=False)
        self.lateral_proj = nn.Conv2d(c2_dim, fusion_dim, kernel_size=1, bias=False)
        self.post_fusion_conv = nn.Sequential(
            DepthwiseSeparableConv(
                fusion_dim, fusion_dim, kernel_size=3, padding=1,
                dropblock_prob=dropblock_prob
            ),
            nn.GroupNorm(num_groups=8, num_channels=fusion_dim),
            nn.ReLU(inplace=True)
        )
        self.drop_path = DropPath(drop_prob=droppath_prob)
        self.out_proj = nn.Conv2d(fusion_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, f_shallow, f_deep):
        deep_proj = self.top_down_proj(f_deep)
        deep_upsampled = F.interpolate(
            deep_proj, size=f_shallow.shape[-2:], mode='bilinear', align_corners=False
        )
        shallow_proj = self.lateral_proj(f_shallow)
        fused = shallow_proj + deep_upsampled
        low_dim = self.post_fusion_conv(fused)
        low_dim = self.drop_path(low_dim)
        out = self.out_proj(low_dim)
        return out

class CoAtNetSideViTClassifier_2(nn.Module):
    def __init__(self, side_vit1, side_vit2, side_vit_cnn, cfg: Any, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            'coatnet_0_rw_224', pretrained=pretrained, features_only=True
        )
        # freeze except last two blocks
        for name, param in self.backbone.named_parameters():
            param.requires_grad = any(f'blocks.{i}' in name for i in (2, 3))

        feat = self.backbone.feature_info
        c2, c3, c4 = feat[2]['num_chs'], feat[3]['num_chs'], feat[4]['num_chs']
        in_ch = cfg.dataset.image_channel_num
        num_classes = 2
        fusion_dim = getattr(cfg, 'fpn_fusion_dim', 64)

        self.fpn_fusion = LightweightFPNFusion(
            c2, c3, fusion_dim, out_dim=in_ch,
            droppath_prob=0.1, dropblock_prob=0.1
        )
        self.se1 = SEBlock(channel=in_ch)
        self.se2 = SEBlock(channel=in_ch)
        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1, bias=False)
        # Use GroupNorm + Dropout for side inputs
        self.dropout_sv1 = nn.Dropout2d(p=0.3)
        self.dropout_sv2 = nn.Dropout2d(p=0.3)

        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        # Classification head with Dropout and weight decay friendly LayerNorm
        hidden_dim = getattr(cfg, 'mlp_hidden_dim', 12)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, K_value=None, Q_value=None):
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        f2, f3, f4 = self.backbone(x_resized)[2:]

        sv1 = self.fpn_fusion(f2, f3)
        sv1 = F.interpolate(sv1, size=(128, 128), mode='bilinear', align_corners=False)
        sv1 = self.se1(sv1)
        sv1 = self.dropout_sv1(sv1)

        sv2 = self.proj_sv2(f4)
        sv2 = F.interpolate(sv2, size=(128, 128), mode='bilinear', align_corners=False)
        sv2 = self.se2(sv2)
        sv2 = self.dropout_sv2(sv2)

        out1 = self.sidevit1(sv1, K_value, Q_value)
        out2 = self.sidevit2(sv2, K_value, Q_value)
        out3 = self.side_vit_cnn(x, K_value, Q_value)

        combined = torch.cat([out1, out2, out3], dim=1)
        logits = self.cls_head(combined)
        return logits
