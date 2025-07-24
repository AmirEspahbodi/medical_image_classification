import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from typing import Any, Tuple
from timm.models.layers import DropPath, create_conv2d
from timm.layers.std_conv import StdConv2d
from timm.layers import DropBlock2d, Mlp

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

class CoAtNetSideViTClassifier_V3(nn.Module):
    def __init__(
        self,
        side_vit1,
        side_vit2,
        side_vit_cnn,
        cfg: Any,
        pretrained: bool = True,
        # NEW: Regularization parameters
        drop_path_rate: float = 0.2,
        drop_block_p: float = 0.2,
        mlp_dropout_p: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        
        # --- Backbone: CoAtNet with Stochastic Depth (DropPath) ---
        # ✨ NEW: Added drop_path_rate for strong backbone regularization.
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
        num_classes = 2 # Assuming binary classification

        # --- Input Processing for Side-ViTs with DropBlock ---
        
        # 1. For Side-ViT 1 (Multi-scale FPN Input)
        fusion_dim = getattr(self.cfg, 'fpn_fusion_dim', 64)
        self.fpn_fusion = LightweightFPNFusion(c2_dim=c2_dim, c3_dim=c3_dim, fusion_dim=fusion_dim, out_dim=in_ch)
        self.se_block_sv1 = SEBlock(channel=in_ch)
        # ✨ NEW: Replaced Dropout2d with more effective DropBlock2d
        self.dropout_sv1 = DropBlock2d(drop_prob=drop_block_p, block_size=7)

        # 2. For Side-ViT 2 (Single-scale Input)
        self.proj_sv2 = nn.Conv2d(c4_dim, in_ch, kernel_size=1, bias=False)
        self.se_block_sv2 = SEBlock(channel=in_ch)
        # ✨ NEW: Replaced Dropout2d with more effective DropBlock2d
        self.dropout_sv2 = DropBlock2d(drop_prob=drop_block_p, block_size=7)


        # --- Side-ViT Ensembles (Unchanged) ---
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        # --- Fortified Classification Head ---
        # ✨ NEW: Added BatchNorm1d and a higher Dropout rate for a more robust classifier.
        hidden_dim = getattr(self.cfg, 'mlp_hidden_dim', 32) # Increased hidden dim slightly
        # Assuming each side-vit outputs 2 logits for binary classification. 2+2+2 = 6
        mlp_in_features = num_classes * 3 
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Stabilizes and regularizes
            nn.ReLU(inplace=True),
            nn.Dropout(p=mlp_dropout_p), # Strong regularization before the final layer
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, K_value=None, Q_value=None) -> torch.Tensor:
        # Backbone forward pass (Unchanged)
        x_backbone = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.backbone(x_backbone)
        f2, f3, f4 = features[2], features[3], features[4]

        # Side-ViT 1 Input Path (Unchanged logic, but uses new DropBlock module)
        sv1_in = self.fpn_fusion(f_shallow=f2, f_deep=f3)
        sv1_in = F.interpolate(sv1_in, size=(128, 128), mode='bilinear', align_corners=False)
        sv1_in = self.se_block_sv1(sv1_in)
        sv1_in = self.dropout_sv1(sv1_in)

        # Side-ViT 2 Input Path (Unchanged logic, but uses new DropBlock module)
        sv2_in = self.proj_sv2(f4)
        sv2_in = F.interpolate(sv2_in, size=(128, 128), mode='bilinear', align_corners=False)
        sv2_in = self.se_block_sv2(sv2_in)
        sv2_in = self.dropout_sv2(sv2_in)

        # Side-ViT-CNN Input (Unchanged)
        sv3_in = x

        # Forward through Side-ViTs (Unchanged)
        out1 = self.sidevit1(sv1_in, K_value, Q_value)
        out2 = self.sidevit2(sv2_in, K_value, Q_value)
        out3 = self.side_vit_cnn(sv3_in, K_value, Q_value)

        # Final Combination and Classification (Unchanged logic)
        combined = torch.cat([out1, out2, out3], dim=1)
        logits = self.mlp(combined)
        return logits
