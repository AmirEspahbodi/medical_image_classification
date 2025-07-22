import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from typing import Any, Tuple

# ---- Novel Pre-Side-ViT Modules ----
class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: Tuple[int, ...] = (1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for rate in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(len(rates) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return self.project(torch.cat(feats, dim=1))


class FeatureAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # use small ASPP for Q/K/V pooling
        self.qkv_pool = ASPP(channels, channels // 4 or 1, rates=(1, 2, 3))
        self.project = nn.Sequential(
            nn.Conv2d(channels // 4 or 1, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.project(self.qkv_pool(x))
        return x * attn


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 7, stride: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x)


# ---- Enhanced Pre-Side-ViT Encoder ----
class EnhancedPreSideViT(nn.Module):
    def __init__(
        self,
        cfg: Any,
        resnet_variant: str = 'resnet50',
        pretrained: bool = True
    ):
        super().__init__()
        # Load ResNet backbone
        variants = {
            'resnet18': (models.resnet18,  [64, 128, 256, 512]),
            'resnet50': (models.resnet50, [256, 512, 1024, 2048]),
            'resnet101': (models.resnet101, [256, 512, 1024, 2048])
        }
        if resnet_variant not in variants:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")
        backbone_fn, channels = variants[resnet_variant]
        backbone = backbone_fn(pretrained=pretrained)

        # Freeze early layers
        for name, param in backbone.named_parameters():
            if 'layer3' not in name and 'layer4' not in name:
                param.requires_grad = False

        # Stem and layers
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # FPN laterals
        C = cfg.dataset.image_channel_num
        self.l2 = nn.Conv2d(channels[1], C, kernel_size=1)
        self.l3 = nn.Conv2d(channels[2], C, kernel_size=1)
        self.l4 = nn.Conv2d(channels[3], C, kernel_size=1)

        # Novel enhancement modules
        self.aspp2 = ASPP(C, C);
        self.aspp3 = ASPP(C, C);
        self.aspp4 = ASPP(C, C)
        self.fam2  = FeatureAttention(C);
        self.fam3  = FeatureAttention(C);
        self.fam4  = FeatureAttention(C)

        # Overlapping patch embed
        self.patch2 = OverlapPatchEmbed(C, C, patch_size=7, stride=2)
        self.patch3 = OverlapPatchEmbed(C, C, patch_size=7, stride=2)
        self.patch4 = OverlapPatchEmbed(C, C, patch_size=7, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ResNet->FPN
        x0 = self.stem(x)
        f2 = self.layer2(self.layer1(x0))
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        p4 = self.l4(f4)
        p3 = self.l3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode='nearest')
        p2 = self.l2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode='nearest')

        # ASPP + FAM
        e2 = self.fam2(self.aspp2(p2))
        e3 = self.fam3(self.aspp3(p3))
        e4 = self.fam4(self.aspp4(p4))

        # Patch embed + resize to 128x128
        t1 = F.interpolate(self.patch2(e2), size=(128, 128), mode='bilinear', align_corners=False)
        t2 = F.interpolate(self.patch3(e3), size=(128, 128), mode='bilinear', align_corners=False)
        t3 = F.interpolate(self.patch4(e4), size=(128, 128), mode='bilinear', align_corners=False)

        return t1, t2, t3


# ---- Full Classifier Pipeline ----
class ResNetSideViTClassifier_MLP_CNNVIT3(nn.Module):
    def __init__(
        self,
        side_vit1: nn.Module,
        side_vit2: nn.Module,
        side_vit_cnn: nn.Module,
        cfg: Any,
        resnet_variant: str = 'resnet50',
        pretrained: bool = True
    ):
        super().__init__()
        # Enhanced pre-sidevit encoder
        self.feature_refinery = EnhancedPreSideViT(cfg, resnet_variant, pretrained)

        # Side-ViT modules (black-box)
        self.sidevit1    = side_vit1
        self.sidevit2    = side_vit2
        self.sidevit_cnn = side_vit_cnn

        # Fusion and classification head
        side_dim  = cfg.model.side_output_dim  # e.g., 2
        total_dim = side_dim * 3               # e.g., 6
        self.se = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2 or 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(total_dim // 2 or 1, total_dim, bias=False),
            nn.Sigmoid()
        )
        hidden = getattr(cfg, 'mlp_hidden_dim', 16)
        self.norm = nn.LayerNorm(total_dim)
        self.dropout = nn.Dropout(getattr(cfg, 'dropout', 0.5))
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, cfg.dataset.num_classes)
        )

    def forward(self, x: torch.Tensor, K_value, Q_value) -> torch.Tensor:
        # Produce refined feature maps
        feat1, feat2, feat3 = self.feature_refinery(x)

        # Side-ViT predictions
        out1 = self.sidevit1(feat1, K_value, Q_value)
        out2 = self.sidevit2(feat2, K_value, Q_value)
        out3 = self.sidevit_cnn(feat3, K_value, Q_value)

        # Fuse with SE attention
        combined = torch.cat([out1, out2, out3], dim=1)
        attn = self.se(combined)
        fused = combined * attn

        # Classify
        x = self.norm(fused)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


def se_block(channels: int, reduction: int = 2) -> nn.Module:
    inner_dim = max(channels // reduction, 1)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(channels, inner_dim, kernel_size=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_dim, channels, kernel_size=1, bias=False),
        nn.Sigmoid()
    )

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




class FusionAdapter(nn.Module):
    """
    Lightweight adapter that fuses two feature maps via a 1x1 conv and self-attention.
    """
    def __init__(self, in_c1: int, in_c2: int, fusion_dim: int = 256, num_heads: int = 8):
        super().__init__()
        # Project concatenated features to fusion_dim
        self.project = nn.Conv2d(in_c1 + in_c2, fusion_dim, kernel_size=1)
        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        # feat1, feat2: [B, C, H, W]
        # Align spatial dims
        if feat1.shape[-2:] != feat2.shape[-2:]:
            feat2 = F.interpolate(feat2, size=feat1.shape[-2:], mode='bilinear', align_corners=False)
        # Concatenate and project
        x = torch.cat([feat1, feat2], dim=1)           # [B, C1+C2, H, W]
        x = self.project(x)                            # [B, fusion_dim, H, W]
        B, C, H, W = x.shape
        # Flatten to sequence
        seq = x.flatten(2).transpose(1, 2)             # [B, H*W, fusion_dim]
        # Self-attention
        attn_out, _ = self.mha(seq, seq, seq)         # [B, H*W, fusion_dim]
        # Residual + Norm
        seq = self.norm(seq + attn_out)               # [B, H*W, fusion_dim]
        # Reshape back
        out = seq.transpose(1, 2).view(B, C, H, W)     # [B, fusion_dim, H, W]
        return out

class ResNetSideViTClassifier_MLP_CNNVIT(nn.Module):
    def __init__(
        self,
        side_vit1: nn.Module,
        side_vit2: nn.Module,
        side_vit_cnn: nn.Module,
        cfg: Any,
        backbone: str = 'coatnet0_rw_224',
        pretrained: bool = True
    ):
        super().__init__()
        # 1) Load a hybrid CNN-Transformer backbone
        #    Using timm's features_only to get intermediate feature maps
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        feat_dims = self.backbone.feature_info.channels()  # e.g. [64, 128, 320, 512]
        c1, c2, c3, c4 = feat_dims

        # 2) Freeze only the earliest stage(s)
        for name, module in self.backbone.named_children():
            if name in ['stem', 'stage1']:
                for p in module.parameters(): p.requires_grad = False

        # 3) Fusion adapters for linking feature maps
        self.fuse23 = FusionAdapter(c2, c3, fusion_dim=c3)
        self.fuse4  = FusionAdapter(c4, c4, fusion_dim=c4)

        # 4) Project fused features to Side-ViT input channels
        in_ch = cfg.dataset.image_channel_num
        self.proj_sv1 = nn.Conv2d(c3, in_ch, kernel_size=1)
        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1)

        # 5) Side-ViT modules (black boxes)
        self.sidevit1    = side_vit1
        self.sidevit2    = side_vit2
        self.sidevit_cnn = side_vit_cnn

        # 6) Final classification head
        total_dim = 6
        hidden_dim = getattr(cfg, 'mlp_hidden_dim', 16)
        self.head = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, cfg.dataset.num_classes)
        )

    def forward(self, x: torch.Tensor, K_value, Q_value) -> torch.Tensor:
        # Extract multi-scale features
        feats = self.backbone(x)   # list: [f1, f2, f3, f4]
        f2, f3, f4 = feats[1], feats[2], feats[3]

        # Fuse layer2 & layer3 for Side-ViT1
        f23 = self.fuse23(f2, f3)  # [B, c3, H2, W2]
        sv1_in = self.proj_sv1(f23)
        sv1_in = F.interpolate(sv1_in, size=(128, 128), mode='bilinear', align_corners=False)

        # Fuse layer4 with itself (identity) for Side-ViT2
        f4_fused = self.fuse4(f4, f4)
        sv2_in = self.proj_sv2(f4_fused)
        sv2_in = F.interpolate(sv2_in, size=(128, 128), mode='bilinear', align_corners=False)

        # Side-ViT predictions
        out1 = self.sidevit1(sv1_in, K_value, Q_value)
        out2 = self.sidevit2(sv2_in, K_value, Q_value)
        out3 = self.sidevit_cnn(x,   K_value, Q_value)

        # Concatenate and classify
        combined = torch.cat([out1, out2, out3], dim=-1)
        logits = self.head(combined)
        return logits

