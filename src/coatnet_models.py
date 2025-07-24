import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from typing import Any, Tuple, Optional
from timm.models.layers import DropPath, create_conv2d
from timm.layers.std_conv import StdConv2d
from timm.layers import DropBlock2d, Mlp

class CoAtNetSideViTClassifier_1(nn.Module):
    """
    Revised classifier with a focus on parameter reduction.

    Changes:
    1.  **Removed Attention Fusion:** The attention mechanism is replaced.
    2.  **Simple & Efficient MLP Head:** A lightweight MLP now serves as the
        classification head, directly processing the concatenated features.
        This significantly reduces parameters in the head.
    """
    def __init__(
        self,
        side_vit1: nn.Module,
        side_vit2: nn.Module,
        side_vit_cnn: nn.Module,
        cfg: Any,
        pretrained: bool = True,
    ):
        super().__init__()
        # --- Regularization Hyperparameters ---
        self.drop_path_rate = getattr(cfg, 'drop_path_rate', 0.1)
        self.drop_block_p = getattr(cfg, 'drop_block_p', 0.3)
        self.head_dropout = getattr(cfg, 'head_dropout', 0.2)

        # --- Backbone: CoAtNet with Stochastic Depth ---
        self.backbone = timm.create_model(
            'coatnet_0_rw_224',
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=self.drop_path_rate # Retained for regularization
        )
        for name, param in self.backbone.named_parameters():
            param.requires_grad = 'block3' in name or 'block4' in name

        # --- Model Parameters ---
        c2, c3, c4 = 192, 384, 768
        in_ch = cfg.dataset.image_channel_num
        num_classes = cfg.dataset.num_classes

        # --- Adapters with DropBlock ---
        self.proj_sv1 = nn.Conv2d(c2 + c3, in_ch, kernel_size=1, bias=False)
        self.adapt_sv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            DropBlock2d(self.drop_block_p, block_size=7) # Retained for regularization
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
        
        # 🔥 Simple, low-parameter MLP Head
        vit_out_features = 2  # Assuming 2 features per side-ViT
        total_vit_features = vit_out_features * 3 # 6 total features
        
        # A small hidden dimension reduces parameters significantly
        mlp_hidden_dim = getattr(cfg, 'mlp_hidden_dim', 12) 

        self.mlp = nn.Sequential(
            nn.Linear(total_vit_features, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(mlp_hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, K_value=None, Q_value=None) -> torch.Tensor:
        # 1) Backbone Feature Extraction
        x_backbone = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.backbone(x_backbone)
        f2, f3, f4 = features[2], features[3], features[4]

        # 2) Prepare inputs for Side-ViTs
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        feats23 = torch.cat([f2, f3_up], dim=1)
        sv1_in = self.adapt_sv1(self.proj_sv1(feats23))
        
        sv2_in = self.adapt_sv2(self.proj_sv2(f4))

        # 3) Forward through Side-ViTs
        vit_out1 = self.sidevit1(F.interpolate(sv1_in, size=(128, 128), mode='bilinear', align_corners=False), K_value, Q_value)
        vit_out2 = self.sidevit2(F.interpolate(sv2_in, size=(128, 128), mode='bilinear', align_corners=False), K_value, Q_value)
        vit_out3 = self.side_vit_cnn(x, K_value, Q_value)

        # 4) Simple Concatenation and MLP Classification
        combined = torch.cat([vit_out1, vit_out2, vit_out3], dim=1)
        logits = self.mlp(combined)
        
        return logits



## -----------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropBlock2d
from typing import Any

# --- Helper Modules (Unchanged) ---
# DepthwiseSeparableConv, SEBlock, and LightweightFPNFusion remain the same.
# For brevity, their code is omitted here but should be included in your file.

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

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

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LightweightFPNFusion(nn.Module):
    def __init__(self, c2_dim, c3_dim, fusion_dim, out_dim):
        super().__init__()
        self.top_down_proj = nn.Conv2d(c3_dim, fusion_dim, kernel_size=1, bias=False)
        self.lateral_proj = nn.Conv2d(c2_dim, fusion_dim, kernel_size=1, bias=False)
        self.post_fusion_conv = nn.Sequential(
            DepthwiseSeparableConv(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True)
        )
        self.out_proj = nn.Conv2d(fusion_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, f_shallow, f_deep):
        deep_proj = self.top_down_proj(f_deep)
        deep_upsampled = F.interpolate(deep_proj, size=f_shallow.shape[-2:], mode='bilinear', align_corners=False)
        shallow_proj = self.lateral_proj(f_shallow)
        fused = shallow_proj + deep_upsampled
        fused_low_dim = self.post_fusion_conv(fused)
        out = self.out_proj(fused_low_dim)
        return out

# --- Main Model: Enhanced for Regularization ---

class CoAtNetSideViTClassifier_2(nn.Module):
    def __init__(
        self,
        side_vit1,
        side_vit2,
        side_vit_cnn,
        cfg: Any,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
        drop_block_p: float = 0.2,
    ):
        super().__init__()
        self.cfg = cfg
        
        # --- Backbone with DropPath (Unchanged) ---
        self.backbone = timm.create_model(
            'coatnet_0_rw_224', 
            pretrained=pretrained, 
            features_only=True,
            drop_path_rate=drop_path_rate
        )
        
        # --- Fine-tuning Strategy (Unchanged) ---
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if any([f'blocks.{i}' in name for i in (2, 3)]):
                param.requires_grad = True

        # --- Channel Dimensions (Unchanged) ---
        feature_info = self.backbone.feature_info
        c2_dim = feature_info[2]['num_chs']
        c3_dim = feature_info[3]['num_chs']
        c4_dim = feature_info[4]['num_chs']
        
        in_ch = self.cfg.dataset.image_channel_num
        num_classes = self.cfg.dataset.num_classes
        
        # --- Input Processing for Side-ViTs ---
        
        # 1. FPN Fusion for Side-ViT 1 (Unchanged)
        fusion_dim = getattr(self.cfg, 'fpn_fusion_dim', 64)
        self.fpn_fusion = LightweightFPNFusion(c2_dim=c2_dim, c3_dim=c3_dim, fusion_dim=fusion_dim, out_dim=in_ch)

        # ✨ NEW: Factorized Projection for Side-ViT 2
        # Instead of a single Conv2d(c4_dim, in_ch), we factorize it to reduce parameters.
        # This projects down to a small bottleneck before projecting up to the target channel size.
        bottleneck_dim = getattr(self.cfg, 'proj_bottleneck_dim', 32)
        self.proj_sv2 = nn.Sequential(
            nn.Conv2d(c4_dim, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, in_ch, kernel_size=1, bias=False),
        )

        # ✨ NEW: Shared Modules for Regularization
        # Use the same SE and DropBlock modules for both paths to reduce parameters.
        self.shared_se_block = SEBlock(channel=in_ch)
        self.shared_drop_block = DropBlock2d(drop_prob=drop_block_p, block_size=7)

        # --- Side-ViT Ensembles (Unchanged) ---
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        hidden_dim = getattr(self.cfg, 'mlp_hidden_dim', 16) # Increased hidden dim slightly
        # Assuming each side-vit outputs 2 logits for binary classification. 2+2+2 = 6
        mlp_in_features = num_classes * 3 
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Stabilizes and regularizes
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2), # Strong regularization before the final layer
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, K_value=None, Q_value=None) -> torch.Tensor:
        x_backbone = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.backbone(x_backbone)
        f2, f3, f4 = features[2], features[3], features[4]

        # --- Side-ViT 1 Input ---
        sv1_in = self.fpn_fusion(f_shallow=f2, f_deep=f3)
        sv1_in = F.interpolate(sv1_in, size=(128, 128), mode='bilinear', align_corners=False)
        sv1_in = self.shared_se_block(sv1_in) # Using shared module
        sv1_in = self.shared_drop_block(sv1_in) # Using shared module

        # --- Side-ViT 2 Input ---
        sv2_in = self.proj_sv2(f4) # Using factorized projection
        sv2_in = F.interpolate(sv2_in, size=(128, 128), mode='bilinear', align_corners=False)
        sv2_in = self.shared_se_block(sv2_in) # Using shared module
        sv2_in = self.shared_drop_block(sv2_in) # Using shared module

        # --- Side-ViT-CNN Input ---
        sv3_in = x

        # --- Forward through Side-ViTs ---
        out1 = self.sidevit1(sv1_in, K_value, Q_value)
        out2 = self.sidevit2(sv2_in, K_value, Q_value)
        out3 = self.side_vit_cnn(sv3_in, K_value, Q_value)

        # Final Combination and Classification (Unchanged logic)
        combined = torch.cat([out1, out2, out3], dim=1)
        logits = self.mlp(combined)
        return logits
