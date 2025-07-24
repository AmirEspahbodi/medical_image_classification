import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from typing import Any, Tuple
from timm.models.layers import DropPath, create_conv2d
from timm.layers.std_conv import StdConv2d

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import DropPath
from timm.layers.std_conv import StdConv2d
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import DropBlock2d, Mlp
from typing import Any, Optional

# --- Helper Module for Attention-based Fusion ---
class AttentionFusion(nn.Module):
    """
    Learns to weigh the outputs of the three side models.
    This allows the model to dynamically focus on the most informative stream,
    making it more robust and less prone to overfitting on spurious features
    from a single branch.
    """
    def __init__(self, in_features: int = 2, num_streams: int = 3, hidden_dim: Optional[int] = None):
        super().__init__()
        self.num_streams = num_streams
        self.in_features_per_stream = in_features
        total_features = in_features * num_streams # e.g., 2 * 3 = 6
        hidden_dim = hidden_dim or total_features * 2

        # Attention mechanism to compute weights for each stream
        self.attention_net = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_streams),
            nn.Softmax(dim=-1)
        )

    def forward(self, combined_tensor: torch.Tensor) -> torch.Tensor:
        # combined_tensor shape: [batch, total_features]
        # Calculate attention weights
        # attn_weights shape: [batch, num_streams]
        attn_weights = self.attention_net(combined_tensor)

        # Reshape for weighted sum
        # Reshape combined_tensor to [batch, num_streams, features_per_stream]
        reshaped_tensor = combined_tensor.view(-1, self.num_streams, self.in_features_per_stream)
        # Reshape weights to [batch, num_streams, 1] for broadcasting
        attn_weights = attn_weights.unsqueeze(-1)

        # Apply weights and sum
        # weighted_features shape: [batch, num_streams, features_per_stream]
        weighted_features = reshaped_tensor * attn_weights
        # Fused features shape: [batch, features_per_stream]
        fused_output = weighted_features.sum(dim=1)

        return fused_output

class CoAtNetSideViTClassifier_1(nn.Module):
    def __init__(
        self,
        side_vit1: nn.Module,
        side_vit2: nn.Module,
        side_vit_cnn: nn.Module,
        cfg: Any,
        pretrained: bool = True,
    ):
        super().__init__()
        # --- Configurable Hyperparameters for Regularization ---
        self.drop_path_rate = getattr(cfg, 'drop_path_rate', 0.1)
        self.drop_block_p = getattr(cfg, 'drop_block_p', 0.3)
        self.head_dropout = getattr(cfg, 'head_dropout', 0.5)

        # --- Backbone: CoAtNet with Stochastic Depth ---
        self.backbone = timm.create_model(
            'coatnet_0_rw_224',
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=self.drop_path_rate 
        )
        
        # Fine-tuning only the later stages remains a good strategy
        for name, param in self.backbone.named_parameters():
            param.requires_grad = 'block3' in name or 'block4' in name

        # --- Model Parameters ---
        c2, c3, c4 = 192, 384, 768
        in_ch = cfg.dataset.image_channel_num
        num_classes = cfg.dataset.num_classes

        # --- Projection + Adapter with DropBlock ---
        self.proj_sv1 = nn.Conv2d(c2 + c3, in_ch, kernel_size=1, bias=False)
        self.adapt_sv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(), 
            DropBlock2d(self.drop_block_p, block_size=7) 
        )

        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1, bias=False)
        self.adapt_sv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            DropBlock2d(self.drop_block_p, block_size=7)
        )

        # --- Side-ViT Ensembles ---
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn
        
        vit_out_features = 2 
        total_vit_features = vit_out_features * 3

        self.attention_fusion = AttentionFusion(
            in_features=vit_out_features, 
            num_streams=3
        )
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(vit_out_features), 
            nn.Linear(vit_out_features, num_classes)
        )

    def forward(self, x: torch.Tensor, K_value=None, Q_value=None) -> torch.Tensor:
        # 1) Preprocess for backbone
        x_backbone = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.backbone(x_backbone)
        f2, f3, f4 = features[2], features[3], features[4]

        # 2) Side-ViT-1 input
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        feats23 = torch.cat([f2, f3_up], dim=1)
        sv1_in = self.proj_sv1(feats23)
        sv1_in = self.adapt_sv1(F.interpolate(sv1_in, size=(128, 128), mode='bilinear', align_corners=False))

        # 3) Side-ViT-2 input
        sv2_in = self.proj_sv2(f4)
        sv2_in = self.adapt_sv2(F.interpolate(sv2_in, size=(128, 128), mode='bilinear', align_corners=False))
        
        # 4) Forward through Side-ViTs
        vit_out1 = self.sidevit1(sv1_in, K_value, Q_value)
        vit_out2 = self.sidevit2(sv2_in, K_value, Q_value)
        vit_out3 = self.side_vit_cnn(x, K_value, Q_value) # Use original image 'x'

        # 5) Fuse and Classify
        combined = torch.cat([vit_out1, vit_out2, vit_out3], dim=1) # [batch, 6]
        
        # Apply attention-based fusion
        fused_features = self.attention_fusion(combined) # [batch, 2]
        
        # Apply final classification head
        logits = self.mlp(fused_features) # [batch, num_classes]
        
        return logits



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
