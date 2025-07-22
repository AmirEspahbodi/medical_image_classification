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


def make_se_block(channels: int, reduction: int = 16) -> nn.Module:
    """
    Squeeze-and-Excitation block to recalibrate channel-wise features.
    """
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(channels, channels // reduction, 1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels // reduction, channels, 1, bias=True),
        nn.Sigmoid()
    )


class CoAtNetSideViTClassifier_MLP_CNNVIT(nn.Module):
    def __init__(
        self,
        side_vit1: nn.Module,
        side_vit2: nn.Module,
        side_vit_cnn: nn.Module,
        cfg: Any,
        pretrained: bool = True,
    ):
        super().__init__()
        # Load CoAtNet backbone with feature stages
        self.backbone = timm.create_model(
            'coatnet_0_rw_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4)
        )

        # Get channel dims for stages
        c2, c3, c4 = 192, 384, 768
        in_ch = cfg.dataset.image_channel_num

        # --- Fine-tune selected backbone blocks ---
        # Unfreeze stage 2, 3 and 4 layers
        for idx, stage in enumerate(self.backbone.stages):
            if idx in [2, 3, 4]:  # stages numbering may vary
                for param in stage.parameters():
                    param.requires_grad = True
            else:
                for param in stage.parameters():
                    param.requires_grad = False

        # Projection + SE + Norm for Side-ViT inputs
        # Side-ViT1 input = concat of f2 + f3
        self.proj_sv1 = nn.Conv2d(c2 + c3, in_ch, kernel_size=1)
        self.se_sv1 = make_se_block(in_ch)
        self.norm_sv1 = nn.GroupNorm(8, in_ch)
        self.drop_sv1 = nn.Dropout2d(p=0.1)

        # Side-ViT2 input = f4
        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1)
        self.se_sv2 = make_se_block(in_ch)
        self.norm_sv2 = nn.GroupNorm(8, in_ch)
        self.drop_sv2 = nn.Dropout2d(p=0.1)

        # Side-ViT modules
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        # Final fusion and classification
        fusion_dim = 6
        hidden_dim = 12
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, cfg.dataset.num_classes)
        )

    def forward(self, x: torch.Tensor, K_value=None, Q_value=None) -> torch.Tensor:
        # Resize for backbone
        x_back = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.backbone(x_back)
        f2, f3, f4 = features  # shapes: [B, C2, H2, W2], etc.

        # Side-ViT1 path
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        feats23 = torch.cat([f2, f3_up], dim=1)
        sv1 = self.proj_sv1(feats23)
        se1 = self.se_sv1(sv1) * sv1
        sv1 = self.norm_sv1(se1)
        sv1 = self.drop_sv1(sv1)
        sv1 = F.interpolate(sv1, size=(128, 128), mode='bilinear', align_corners=False)

        # Side-ViT2 path
        sv2 = self.proj_sv2(f4)
        se2 = self.se_sv2(sv2) * sv2
        sv2 = self.norm_sv2(se2)
        sv2 = self.drop_sv2(sv2)
        sv2 = F.interpolate(sv2, size=(128, 128), mode='bilinear', align_corners=False)

        # Obtain side-ViT outputs
        out1 = self.sidevit1(sv1, K_value, Q_value)
        out2 = self.sidevit2(sv2, K_value, Q_value)
        out3 = self.side_vit_cnn(x, K_value, Q_value)

        # Fuse and classify
        combined = torch.cat([out1, out2, out3], dim=1)
        logits = self.classifier(combined)
        return logits

