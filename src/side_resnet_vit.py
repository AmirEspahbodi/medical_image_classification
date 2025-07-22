import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from typing import Any, Tuple


class ResNetSideViTClassifier_MLP_CNNVIT2(nn.Module):
    def __init__(
        self,
        side_vit1: nn.Module,
        side_vit2: nn.Module,
        side_vit_cnn: nn.Module,
        cfg: Any,
        resnet_variant: str = 'resnet50',
        pretrained: bool = True,
    ):
        super().__init__()
        # Backbone selection
        variants = {'resnet18': (models.resnet18,  [64, 128, 256, 512]),
                    'resnet50': (models.resnet50, [256, 512, 1024, 2048]),
                    'resnet101': (models.resnet101, [256, 512, 1024, 2048])}
        if resnet_variant not in variants:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")
        backbone_fn, channels = variants[resnet_variant]
        backbone = backbone_fn(pretrained=pretrained)

        # Freeze early layers; fine-tune deeper ones
        for name, param in backbone.named_parameters():
            param.requires_grad = False if 'layer3' not in name and 'layer4' not in name else True

        # Backbone feature extractors
        self.stem  = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  
        self.layer2 = backbone.layer2  
        self.layer3 = backbone.layer3  
        self.layer4 = backbone.layer4  

        # Feature Pyramid: lateral convs to unify channel dims
        self.lateral2 = nn.Conv2d(channels[1], cfg.dataset.image_channel_num, kernel_size=1)
        self.lateral3 = nn.Conv2d(channels[2], cfg.dataset.image_channel_num, kernel_size=1)
        self.lateral4 = nn.Conv2d(channels[3], cfg.dataset.image_channel_num, kernel_size=1)

        # Side-ViT backbones (black-box)
        self.sidevit1     = side_vit1
        self.sidevit2     = side_vit2
        self.sidevit_cnn  = side_vit_cnn

        # Channel attention on combined ViT outputs
        side_dim   = 2
        total_dim  = side_dim * 3  # 6
        self.se    = nn.Sequential(
            nn.Linear(total_dim, max(total_dim // 2, 1), bias=False),  # inner=3
            nn.ReLU(inplace=True),
            nn.Linear(max(total_dim // 2, 1), total_dim, bias=False),
            nn.Sigmoid()
        )

        # Classification head
        # With such small combined_dim, a modest MLP width (e.g. 16) works well
        hidden = getattr(cfg, 'mlp_hidden_dim', 16)
        self.norm     = nn.LayerNorm(total_dim)
        self.dropout  = nn.Dropout(p=getattr(cfg, 'dropout', 0.5))
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, cfg.dataset.num_classes)
        )

        # Initialize new layers
        for m in [self.lateral2, self.lateral3, self.lateral4] + list(self.classifier):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, K_value, Q_value) -> torch.Tensor:
        x0 = self.stem(x)
        f1 = self.layer1(x0);
        f2 = self.layer2(f1);
        f3 = self.layer3(f2);
        f4 = self.layer4(f3);

        p4 = self.lateral4(f4)
        p3 = self.lateral3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode='nearest')
        p2 = self.lateral2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode='nearest')

        feat1 = F.interpolate(p2, size=(128,128), mode='bilinear', align_corners=False)
        feat2 = F.interpolate(p3, size=(128,128), mode='bilinear', align_corners=False)
        feat3 = x

        out1 = self.sidevit1(feat1, K_value, Q_value)
        out2 = self.sidevit2(feat2, K_value, Q_value)
        out3 = self.sidevit_cnn(feat3, K_value, Q_value)

        combined = torch.cat([out1, out2, out3], dim=1)
        # attn     = self.se(combined)
        # fused    = combined * attn

        # x = self.norm(fused)
        # x = self.dropout(x)
        logits = self.classifier(combined) #combined
        return logits

#########################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import timm


class CoAtNetSideViTClassifier_MLP_CNNVIT(nn.Module):
    def __init__(
        self,
        side_vit1,
        side_vit2,
        side_vit_cnn,
        cfg: Any,
        pretrained: bool = True,
    ):
        super().__init__()
        # --- Backbone: CoAtNet ---
        self.backbone = timm.create_model(
            'coatnet_0_rw_224', pretrained=pretrained, features_only=True
        )
        # Freeze all except stage-3 (blocks 2) and stage-4 (blocks 3)
        for name, param in self.backbone.named_parameters():
            if any([f'blocks.{i}' in name for i in (2, 3)]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Channel dims for CoAtNet stages
        c2, c3, c4 = 192, 384, 768
        in_ch = cfg.dataset.image_channel_num  # e.g. 3 for RGB
        num_classes = 2

        # --- Projection + Adapter for Side-ViT inputs ---
        self.proj_sv1 = nn.Conv2d(c2 + c3, in_ch, kernel_size=1, bias=False)
        self.adapt_sv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1, bias=False)
        self.adapt_sv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

        # Side-ViT ensembles (treated as black boxes)
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        # Learnable fusion weights for side-ViT outputs
        self.fusion_logits = nn.Parameter(torch.zeros(3))

        # Final MLP head now matches summed fusion dimension
        hidden_dim = getattr(cfg, 'mlp_hidden_dim', 12)
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, K_value=None, Q_value=None) -> torch.Tensor:
        # 1) Preprocess for backbone
        x_backbone = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.backbone(x_backbone)
        f2, f3, f4 = features[2], features[3], features[4]

        # 2) Side-ViT-1 input (multi-scale fusion)
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        feats23 = torch.cat([f2, f3_up], dim=1)
        sv1_in = self.proj_sv1(feats23)
        sv1_in = self.adapt_sv1(F.interpolate(sv1_in, size=(128, 128), mode='bilinear', align_corners=False))

        # 3) Side-ViT-2 input
        sv2_in = self.proj_sv2(f4)
        sv2_in = self.adapt_sv2(F.interpolate(sv2_in, size=(128, 128), mode='bilinear', align_corners=False))

        # 4) Side-ViT-CNN input (raw image with aug applied upstream if any)
        sv3_in = x

        # 5) Forward through Side-ViTs (black boxes)
        out1 = self.sidevit1(sv1_in, K_value, Q_value)
        out2 = self.sidevit2(sv2_in, K_value, Q_value)
        out3 = self.side_vit_cnn(sv3_in, K_value, Q_value)

        # 6) Adaptive fusion of side outputs (sum with learned weights)
        weights = torch.softmax(self.fusion_logits, dim=0)
        fused = weights[0] * out1 + weights[1] * out2 + weights[2] * out3

        # 7) Final classification
        logits = self.mlp(fused)
        return logits

## -----------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List
import timm

# --- Helper Module: Squeeze-and-Excitation Block ---
# This block allows the network to perform channel-wise feature recalibration.
# It learns to selectively emphasize informative features and suppress less useful ones.
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- Helper Module: Advanced Adapter with Attention ---
# This replaces the simple Conv-BN-ReLU block with a more robust version
# that includes a residual connection and an SEBlock for attention.
class AttentionAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        # Using a residual block structure for more stable training
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se_block = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        # The input 'x' has already been projected to the correct channel dimension
        residual = x
        x = self.conv_block(x)
        x = self.se_block(x)
        x = x + residual # Additive residual connection
        x = self.relu(x)
        x = self.dropout(x)
        return x

# --- Helper Module: FPN-style Feature Fusion ---
# This module creates a richer multi-scale feature map for the first Side-ViT
# by properly merging features from different backbone stages.
class FPNFusion(nn.Module):
    def __init__(self, c2_dim, c3_dim, out_dim):
        super().__init__()
        # 1x1 conv on the deeper feature map
        self.top_down_proj = nn.Conv2d(c3_dim, out_dim, kernel_size=1, bias=False)
        # 1x1 conv on the shallower feature map
        self.lateral_proj = nn.Conv2d(c2_dim, out_dim, kernel_size=1, bias=False)
        # 3x3 conv to reduce aliasing artifacts after fusion
        self.post_fusion_conv = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_shallow, f_deep):
        # f_shallow is the feature from block 2 (f2)
        # f_deep is the feature from block 3 (f3)
        
        # 1. Apply 1x1 conv to the deeper feature map
        deep_proj = self.top_down_proj(f_deep)
        
        # 2. Upsample the projected deep feature to match the shallow feature's size
        deep_upsampled = F.interpolate(deep_proj, size=f_shallow.shape[-2:], mode='bilinear', align_corners=False)
        
        # 3. Apply 1x1 conv to the shallow feature map
        shallow_proj = self.lateral_proj(f_shallow)
        
        # 4. Fuse by element-wise addition
        fused = shallow_proj + deep_upsampled
        
        # 5. Apply a final 3x3 convolution for better feature integration
        fused = self.post_fusion_conv(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)
        
        return fused

class CoAtNetSideViTClassifier_Advanced(nn.Module):
    def __init__(
        self,
        side_vit1,
        side_vit2,
        side_vit_cnn,
        cfg: Any,
        pretrained: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        
        # --- Backbone: CoAtNet ---
        # We create the backbone and specify that we want feature maps from each stage.
        self.backbone = timm.create_model(
            'coatnet_0_rw_224', pretrained=pretrained, features_only=True
        )
        
        # --- Fine-tuning Strategy ---
        # Freeze all backbone parameters initially.
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze the last two stages (blocks 2 and 3) for fine-tuning.
        # These layers capture high-level semantic features that are crucial for the task.
        for name, param in self.backbone.named_parameters():
            if any([f'blocks.{i}' in name for i in (2, 3)]):
                param.requires_grad = True

        # --- Channel Dimensions ---
        # These are the output channel dimensions from the CoAtNet stages we will use.
        # features[2] -> c2_dim, features[3] -> c3_dim, features[4] -> c4_dim
        feature_info = self.backbone.feature_info.get_dicts(keys=[2, 3, 4])
        c2_dim = feature_info[0]['num_chs'] # Typically 192 for coatnet_0
        c3_dim = feature_info[1]['num_chs'] # Typically 384 for coatnet_0
        c4_dim = feature_info[2]['num_chs'] # Typically 768 for coatnet_0
        
        in_ch = self.cfg.dataset.image_channel_num  # e.g., 3 for RGB
        num_classes = self.cfg.dataset.num_classes

        # --- Advanced Input Processing for Side-ViTs ---
        
        # 1. For Side-ViT 1 (Multi-scale FPN Input)
        self.fpn_fusion = FPNFusion(c2_dim=c2_dim, c3_dim=c3_dim, out_dim=in_ch)
        self.adapter_sv1 = AttentionAdapter(in_channels=in_ch, out_channels=in_ch, dropout_rate=0.15)

        # 2. For Side-ViT 2 (Single-scale Input from final stage)
        # Project the high-dimensional feature map to the required input channels for the adapter.
        self.proj_sv2 = nn.Conv2d(c4_dim, in_ch, kernel_size=1, bias=False)
        self.adapter_sv2 = AttentionAdapter(in_channels=in_ch, out_channels=in_ch, dropout_rate=0.15)

        # --- Side-ViT Ensembles (Treated as Black Boxes) ---
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        # --- Fusion and Final Classification Head ---
        
        # Learnable weights to balance the contribution of each Side-ViT's output.
        # Initializing to zeros makes the initial softmax output uniform (1/3 for each).
        self.fusion_logits = nn.Parameter(torch.zeros(3))

        # Final MLP head to process the fused logits.
        hidden_dim = getattr(self.cfg, 'mlp_hidden_dim', 12)
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, K_value=None, Q_value=None) -> torch.Tensor:
        # 1) Preprocess image for backbone (resize to 224x224)
        # This ensures the input dimensions match what the pretrained CoAtNet expects.
        x_backbone = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 2) Extract multi-scale features from the backbone
        features = self.backbone(x_backbone)
        f2, f3, f4 = features[2], features[3], features[4]

        # 3) Prepare input for Side-ViT 1 using FPN Fusion
        sv1_in = self.fpn_fusion(f_shallow=f2, f_deep=f3)
        sv1_in = F.interpolate(sv1_in, size=(128, 128), mode='bilinear', align_corners=False)
        sv1_in = self.adapter_sv1(sv1_in) # Apply attention-based adapter

        # 4) Prepare input for Side-ViT 2
        sv2_in = self.proj_sv2(f4)
        sv2_in = F.interpolate(sv2_in, size=(128, 128), mode='bilinear', align_corners=False)
        sv2_in = self.adapter_sv2(sv2_in) # Apply attention-based adapter

        # 5) Prepare input for Side-ViT-CNN (raw image)
        # No processing needed here as it takes the original input.
        sv3_in = x

        # 6) Forward pass through the three Side-ViTs (black boxes)
        out1 = self.sidevit1(sv1_in, K_value, Q_value)
        out2 = self.sidevit2(sv2_in, K_value, Q_value)
        out3 = self.side_vit_cnn(sv3_in, K_value, Q_value)

        # 7) Adaptively fuse the outputs from the Side-ViTs
        # Softmax ensures the weights sum to 1, providing a weighted average.
        weights = torch.softmax(self.fusion_logits, dim=0)
        fused = weights[0] * out1 + weights[1] * out2 + weights[2] * out3

        # 8) Final classification through the MLP head
        logits = self.mlp(fused)
        
        return logits
