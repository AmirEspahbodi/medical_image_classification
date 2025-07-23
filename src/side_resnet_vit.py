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
class ResNetSideViTClassifier_MLP_CNNVIT(nn.Module):
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
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False if 'layer3' not in name and 'layer4' not in name else True


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
            nn.Dropout2d(p=0.4)
        )

        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1, bias=False)
        self.adapt_sv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.4)
        )

        # Side-ViT ensembles (treated as black boxes)
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        # Final MLP head with stronger dropout and smaller hidden size
        hidden_dim = getattr(cfg, 'mlp_hidden_dim', 12)
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cfg.dataset.num_classes)
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

        # 4) Side-ViT-CNN input (raw image)
        sv3_in = x

        # 5) Forward through Side-ViTs (black boxes)
        vit_out1 = self.sidevit1(sv1_in, K_value, Q_value)
        vit_out2 = self.sidevit2(sv2_in, K_value, Q_value)
        vit_out3 = self.side_vit_cnn(sv3_in, K_value, Q_value)

        # 6) Fixed-average fusion + dropout
        # Combine and classify
        combined = torch.cat([vit_out1, vit_out2, vit_out3], dim=1)  # [batch, 4]
        logits = self.mlp(combined)
        return logits

## -----------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List
import timm

# --- Helper Module: Depthwise Separable Convolution ---
# This layer is a parameter-efficient replacement for a standard Conv2d.
# It splits the convolution into a depthwise (spatial) and a pointwise (channel) operation.
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# --- Helper Module: Squeeze-and-Excitation Block (with Regularization) ---
# This block provides channel-wise attention to re-weight feature maps,
# with dropout for regularization to prevent overfitting.
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

# --- Helper Module: Lightweight FPN-style Feature Fusion ---
# This module reduces parameters by performing the fusion and 3x3 convolution
# in a low-dimensional space before projecting to the final output dimension.
class LightweightFPNFusion(nn.Module):
    def __init__(self, c2_dim, c3_dim, fusion_dim, out_dim):
        super().__init__()
        # Project backbone features into a low-dimensional space
        self.top_down_proj = nn.Conv2d(c3_dim, fusion_dim, kernel_size=1, bias=False)
        self.lateral_proj = nn.Conv2d(c2_dim, fusion_dim, kernel_size=1, bias=False)
        
        # 3x3 convolution now operates on the smaller fusion_dim using a lightweight conv
        self.post_fusion_conv = nn.Sequential(
            DepthwiseSeparableConv(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final projection to the desired output dimension for the Side-ViT
        self.out_proj = nn.Conv2d(fusion_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, f_shallow, f_deep):
        deep_proj = self.top_down_proj(f_deep)
        deep_upsampled = F.interpolate(deep_proj, size=f_shallow.shape[-2:], mode='bilinear', align_corners=False)
        shallow_proj = self.lateral_proj(f_shallow)
        fused = shallow_proj + deep_upsampled
        fused_low_dim = self.post_fusion_conv(fused)
        out = self.out_proj(fused_low_dim)
        return out

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
        
        # --- Backbone: CoAtNet (Unchanged) ---
        self.backbone = timm.create_model(
            'coatnet_0_rw_224', pretrained=pretrained, features_only=True
        )
        
        # --- Fine-tuning Strategy (Unchanged) ---
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if any([f'blocks.{i}' in name for i in (2, 3)]):
                param.requires_grad = True

        # --- Channel Dimensions ---
        # Correctly access feature info directly from the backbone's feature_info attribute.
        # This is a list of dicts, one for each feature map returned by features_only=True.
        feature_info = self.backbone.feature_info
        c2_dim = feature_info[2]['num_chs']
        c3_dim = feature_info[3]['num_chs']
        c4_dim = feature_info[4]['num_chs']
        
        in_ch = self.cfg.dataset.image_channel_num
        num_classes = 2 # Should be 4 for your dataset

        # --- Ultra-Lightweight Input Processing for Side-ViTs ---
        
        # 1. For Side-ViT 1 (Multi-scale FPN Input)
        fusion_dim = getattr(self.cfg, 'fpn_fusion_dim', 64)
        self.fpn_fusion = LightweightFPNFusion(c2_dim=c2_dim, c3_dim=c3_dim, fusion_dim=fusion_dim, out_dim=in_ch)
        self.se_block_sv1 = SEBlock(channel=in_ch)
        self.dropout_sv1 = nn.Dropout2d(p=0.2)

        # 2. For Side-ViT 2 (Single-scale Input from final stage)
        self.proj_sv2 = nn.Conv2d(c4_dim, in_ch, kernel_size=1, bias=False)
        self.se_block_sv2 = SEBlock(channel=in_ch)
        self.dropout_sv2 = nn.Dropout2d(p=0.2)


        # --- Side-ViT Ensembles (Treated as Black Boxes) ---
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        # --- Final Classification Head ---
        # The input dimension of the MLP is num_classes because we are averaging the outputs.
        hidden_dim = getattr(self.cfg, 'mlp_hidden_dim', 8)
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, K_value=None, Q_value=None) -> torch.Tensor:
        x_backbone = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.backbone(x_backbone)
        f2, f3, f4 = features[2], features[3], features[4]

        # Side-ViT 1 Input
        sv1_in = self.fpn_fusion(f_shallow=f2, f_deep=f3)
        sv1_in = F.interpolate(sv1_in, size=(128, 128), mode='bilinear', align_corners=False)
        sv1_in = self.se_block_sv1(sv1_in)
        sv1_in = self.dropout_sv1(sv1_in)


        # Side-ViT 2 Input
        sv2_in = self.proj_sv2(f4)
        sv2_in = F.interpolate(sv2_in, size=(128, 128), mode='bilinear', align_corners=False)
        sv2_in = self.se_block_sv2(sv2_in)
        sv2_in = self.dropout_sv2(sv2_in)


        # Side-ViT-CNN Input
        sv3_in = x

        # Forward through Side-ViTs
        out1 = self.sidevit1(sv1_in, K_value, Q_value)
        out2 = self.sidevit2(sv2_in, K_value, Q_value)
        out3 = self.side_vit_cnn(sv3_in, K_value, Q_value)

        # Fusion by simple averaging
        fused = (out1 + out2 + out3) / 3.0
        
        # Final Classification
        logits = self.mlp(fused)
        
        return logits
