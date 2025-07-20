import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .bridge import FineGrainedPromptTuning
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .bridge import FineGrainedPromptTuning
from typing import Any

class ResNetSideViTClassifier_old(nn.Module):
    def __init__(
        self,
        side_vit: FineGrainedPromptTuning,
        cfg: Any,
        resnet_variant: str = 'resnet18',
        pretrained: bool = True,
    ):
        super().__init__()
        # Load ResNet backbone
        if resnet_variant == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            c2, c3 = 128, 256
        elif resnet_variant == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            c2, c3 = 512, 1024
        elif resnet_variant == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
            c2, c3 = 512, 1024
        else:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")

        # --- Freeze all backbone parameters ---
        for param in backbone.parameters():
            param.requires_grad = False

        # Initial layers (stem + layer1-3)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # 1x1 conv to reduce channels to Side-ViT in_chans
        in_ch = cfg.dataset.image_channel_num
        self.proj_conv = nn.Conv2d(c2 + c3, in_ch, kernel_size=1)

        # Side-ViT
        self.sidevit = side_vit


    def forward(self, x: torch.Tensor, K_value, Q_value) -> torch.Tensor:
        # Ensure backbone is not updating any running stats accidentally
        with torch.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            f2 = self.layer2(x)
            f3 = self.layer3(f2)

            # Align spatial dims
            f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
            feats = torch.cat([f2, f3_up], dim=1)

        feats = self.proj_conv(feats)
        feats = F.interpolate(feats, size=(128, 128), mode='bilinear', align_corners=False)

        # Side-ViT and classification
        vit_out = self.sidevit(feats, K_value, Q_value)

        return vit_out


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .bridge import FineGrainedPromptTuning
from typing import Any

class ResNetSideViTClassifier_old2(nn.Module):
    def __init__(
        self,
        side_vit1: FineGrainedPromptTuning,
        side_vit2: FineGrainedPromptTuning,
        cfg: Any,
        resnet_variant: str = 'resnet18',
        pretrained: bool = True,
    ):
        super().__init__()
        # Load ResNet backbone
        if resnet_variant == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            c1, c2, c3, c4 = 64, 128, 256, 512
        elif resnet_variant == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            c1, c2, c3, c4 = 256, 512, 1024, 2048
        elif resnet_variant == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
            c1, c2, c3, c4 = 256, 512, 1024, 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")

        # Freeze backbone
        for param in backbone.parameters():
            param.requires_grad = False
            
        # Initial layers
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # output channels c1
        self.layer2 = backbone.layer2  # output channels c2
        self.layer3 = backbone.layer3  # output channels c3
        self.layer4 = backbone.layer4  # output channels c4

        # Projection from block1+2 to Side-ViT inputs
        in_ch = cfg.dataset.image_channel_num
        self.proj_sv1 = nn.Conv2d(c1 + c2, in_ch, kernel_size=1)
        self.proj_sv2 = nn.Conv2d(c3 + c4, in_ch, kernel_size=1)

        # Encoder-Decoder feed-forward modules for robust feature blending
        hidden_ff = in_ch * 2
        self.encdec1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ff, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ff, in_ch, kernel_size=1),
        )
        self.encdec2 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ff, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ff, in_ch, kernel_size=1),
        )

        # Side-ViT classifiers
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2

        # MLP head with dropout for regularization
        hidden_dim = getattr(cfg, 'mlp_hidden_dim', 8)
        mlp_dropout = getattr(cfg, 'mlp_dropout', 0.3)
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, cfg.dataset.num_classes)
        )

    def forward(self, x: torch.Tensor, K_value, Q_value) -> torch.Tensor:
        # Extract hierarchical features (backbone frozen)
        with torch.no_grad():
            x0 = self.stem(x)
            f1 = self.layer1(x0)        # block1
            f2 = self.layer2(f1)        # block2
            f3 = self.layer3(f2)        # block3 (unused here)
            f4 = self.layer4(f3)        # block4 (unused here)

        # ----- Build features for Side-ViT-1 -----
        f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        feats12 = torch.cat([f1, f2_up], dim=1)            # [c1+c2, H/4, W/4]
        feats1 = self.proj_sv1(feats12)                    # [in_ch, H/4, W/4]
        feats1 = self.encdec1(feats1)                      # robust blending
        feats1 = F.interpolate(feats1, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Build features for Side-ViT-2 -----
        f4_up = F.interpolate(f4, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        feats34 = torch.cat([f3, f4_up], dim=1)
        feats2 = self.proj_sv2(feats34)                    # [in_ch, H/4, W/4]
        feats2 = self.encdec2(feats2)                      # robust blending
        feats2 = F.interpolate(feats2, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Side-ViT predictions -----
        vit_out1 = self.sidevit1(feats1, K_value, Q_value)
        vit_out2 = self.sidevit2(feats2, K_value, Q_value)

        # Combine and classify
        combined = torch.cat([vit_out1, vit_out2], dim=1)  # [batch, 4]
        logits = self.mlp(combined)
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .bridge import FineGrainedPromptTuning
from typing import Any, Tuple
import math

class EncoderDecoderModule(nn.Module):
    """
    Enhanced Encoder-Decoder module with residual connections and attention
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_channels // 2, out_channels * 2)
        
        # Encoder with bottleneck design
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Residual connection projection if needed
        self.residual_proj = nn.Identity()
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        
        # Spatial attention mechanism
        self.spatial_attention = SpatialAttention()
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        
        # Encoder-decoder pathway
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Apply spatial attention
        decoded = self.spatial_attention(decoded)
        
        # Residual connection with dropout
        output = residual + self.dropout(decoded)
        return F.relu(output, inplace=True)

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism to focus on important regions
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute channel-wise statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention_map = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        return x * attention_map

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion
    """
    def __init__(self, in_channels_list: list, out_channels: int = 256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
            
    def forward(self, x_list: list) -> list:
        # Build pyramid from top to bottom
        results = []
        last_inner = self.inner_blocks[-1](x_list[-1])
        results.append(self.layer_blocks[-1](last_inner))
        
        for idx in range(len(x_list) - 2, -1, -1):
            inner = self.inner_blocks[idx](x_list[idx])
            # Upsample and add
            upsampled = F.interpolate(
                last_inner, size=inner.shape[-2:], 
                mode='bilinear', align_corners=False
            )
            last_inner = inner + upsampled
            results.insert(0, self.layer_blocks[idx](last_inner))
            
        return results

class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive feature fusion with learnable weights
    """
    def __init__(self, num_inputs: int, channels: int):
        super().__init__()
        self.num_inputs = num_inputs
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_inputs, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_inputs, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, features: list) -> torch.Tensor:
        # Concatenate features for weight computation
        concat_features = torch.cat(features, dim=1)
        weights = self.weight_net(concat_features)  # [B, num_inputs, 1, 1]
        
        # Apply weights and sum
        weighted_features = []
        for i, feat in enumerate(features):
            weighted_features.append(feat * weights[:, i:i+1, :, :])
            
        return sum(weighted_features)

class ResNetSideViTClassifier(nn.Module):
    def __init__(
        self,
        side_vit1: FineGrainedPromptTuning,
        side_vit2: FineGrainedPromptTuning,
        cfg: Any,
        resnet_variant: str = 'resnet18',
        pretrained: bool = True,
        use_fpn: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        
        # Load ResNet backbone with channel configurations
        if resnet_variant == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            c1, c2, c3, c4 = 64, 128, 256, 512
        elif resnet_variant == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            c1, c2, c3, c4 = 256, 512, 1024, 2048
        elif resnet_variant == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
            c1, c2, c3, c4 = 256, 512, 1024, 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")

        # Optionally freeze backbone parameters
        freeze_backbone = getattr(cfg, 'freeze_backbone', True)
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            
        # ResNet layers
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # Block 1
        self.layer2 = backbone.layer2  # Block 2
        self.layer3 = backbone.layer3  # Block 3
        self.layer4 = backbone.layer4  # Block 4

        # Configuration
        in_ch = cfg.dataset.image_channel_num
        self.target_size = getattr(cfg, 'sidevit_input_size', (128, 128))
        self.use_fpn = use_fpn
        self.use_attention = use_attention
        
        # Feature Pyramid Network (optional)
        if self.use_fpn:
            self.fpn = FeaturePyramidNetwork([c1, c2, c3, c4], out_channels=256)
            fpn_channels = 256
        else:
            fpn_channels = None

        # Encoder-Decoder modules for feature refinement
        # Path 1: Blocks 1+2 -> SideViT-1
        if self.use_fpn:
            self.encoder_decoder1 = EncoderDecoderModule(
                fpn_channels * 2, in_ch, hidden_dim=fpn_channels
            )
        else:
            self.encoder_decoder1 = EncoderDecoderModule(
                c1 + c2, in_ch, hidden_dim=max(c1 + c2, in_ch * 2)
            )

        # Path 2: Blocks 3+4 -> SideViT-2  
        if self.use_fpn:
            self.encoder_decoder2 = EncoderDecoderModule(
                fpn_channels * 2, in_ch, hidden_dim=fpn_channels
            )
        else:
            self.encoder_decoder2 = EncoderDecoderModule(
                c3 + c4, in_ch, hidden_dim=max(c3 + c4, in_ch * 2)
            )

        # Adaptive feature fusion modules
        if self.use_attention:
            self.adaptive_fusion1 = AdaptiveFeatureFusion(2, c1 if not self.use_fpn else fpn_channels)
            self.adaptive_fusion2 = AdaptiveFeatureFusion(2, c3 if not self.use_fpn else fpn_channels)

        # Side-ViT classifiers
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2

        # Enhanced MLP classifier with more sophisticated architecture
        num_classes = getattr(cfg, 'num_classes', 2)
        vit_output_dim = getattr(cfg, 'vit_output_dim', 2)  # Assuming each ViT outputs 2-dim
        hidden_dim = getattr(cfg, 'mlp_hidden_dim', 64)
        
        self.classifier = nn.Sequential(
            nn.Linear(vit_output_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _resize_and_align_features(self, feat1: torch.Tensor, feat2: torch.Tensor, 
                                 target_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize features to target size with proper alignment"""
        feat1_resized = F.interpolate(
            feat1, size=target_size, mode='bilinear', align_corners=False
        )
        feat2_resized = F.interpolate(
            feat2, size=target_size, mode='bilinear', align_corners=False
        )
        return feat1_resized, feat2_resized

    def forward(self, x: torch.Tensor, K_value, Q_value) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Extract features from ResNet backbone
        with torch.no_grad() if hasattr(self, '_freeze_backbone') else torch.enable_grad():
            x = self.stem(x)
            f1 = self.layer1(x)  # Block 1
            f2 = self.layer2(f1)  # Block 2
            f3 = self.layer3(f2)  # Block 3
            f4 = self.layer4(f3)  # Block 4

        # Apply Feature Pyramid Network if enabled
        if self.use_fpn:
            fpn_features = self.fpn([f1, f2, f3, f4])
            f1, f2, f3, f4 = fpn_features

        # Path 1: Combine blocks 1 and 2 for SideViT-1
        # Resize f1 to match f2 spatial dimensions
        f1_resized = F.interpolate(f1, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.use_attention:
            # Use adaptive fusion for better feature combination
            feats1_combined = self.adaptive_fusion1([f1_resized, f2])
        else:
            # Simple concatenation
            feats1_combined = torch.cat([f1_resized, f2], dim=1)

        # Apply encoder-decoder refinement
        feats1_refined = self.encoder_decoder1(feats1_combined)
        
        # Resize to target size for SideViT-1
        feats1_final = F.interpolate(
            feats1_refined, size=self.target_size, mode='bilinear', align_corners=False
        )

        # Path 2: Combine blocks 3 and 4 for SideViT-2
        # Resize f3 to match f4 spatial dimensions
        f3_resized = F.interpolate(f3, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.use_attention:
            # Use adaptive fusion for better feature combination
            feats2_combined = self.adaptive_fusion2([f3_resized, f4])
        else:
            # Simple concatenation
            feats2_combined = torch.cat([f3_resized, f4], dim=1)

        # Apply encoder-decoder refinement
        feats2_refined = self.encoder_decoder2(feats2_combined)
        
        # Resize to target size for SideViT-2
        feats2_final = F.interpolate(
            feats2_refined, size=self.target_size, mode='bilinear', align_corners=False
        )

        # Pass through SideViT classifiers
        vit_out1 = self.sidevit1(feats1_final, K_value, Q_value)  # Shape: [batch_size, vit_output_dim]
        vit_out2 = self.sidevit2(feats2_final, K_value, Q_value)  # Shape: [batch_size, vit_output_dim]

        # Combine outputs through enhanced MLP classifier
        combined_features = torch.cat([vit_out1, vit_out2], dim=1)
        logits = self.classifier(combined_features)
        
        return logits

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Utility method to extract intermediate feature maps for visualization/analysis
        """
        feature_maps = {}
        
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        feature_maps.update({
            'block1': f1,
            'block2': f2, 
            'block3': f3,
            'block4': f4
        })
        
        if self.use_fpn:
            fpn_features = self.fpn([f1, f2, f3, f4])
            feature_maps.update({
                'fpn_block1': fpn_features[0],
                'fpn_block2': fpn_features[1],
                'fpn_block3': fpn_features[2], 
                'fpn_block4': fpn_features[3]
            })
            
        return feature_maps
