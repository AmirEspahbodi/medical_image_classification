import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .bridge import FineGrainedPromptTuning
from typing import Any


class ResNetSideViTClassifier_1(nn.Module):
    def __init__(
        self,
        side_vit1: FineGrainedPromptTuning,
        side_vit2: FineGrainedPromptTuning,
        side_vit_cnn: FineGrainedPromptTuning,
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
        self.proj_sv1 = nn.Conv2d(c2 + c3, in_ch, kernel_size=1)
        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1)

        # Side-ViT classifiers
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        # MLP head with dropout for regularization
        self.fc = nn.Linear(6, cfg.dataset.num_classes)

    def forward(self, x: torch.Tensor, K_value, Q_value) -> torch.Tensor:
        # Extract hierarchical features (backbone frozen)
        with torch.no_grad():
            x0 = self.stem(x)
            f1 = self.layer1(x0)        # block1
            f2 = self.layer2(f1)        # block2
            f3 = self.layer3(f2)        # block3 (unused here)
            f4 = self.layer4(f3)        # block4 (unused here)

        # ----- Build features for Side-ViT-1 -----
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        feats23 = torch.cat([f2, f3_up], dim=1)
        feats1 = self.proj_sv1(feats23)                 # [in_ch, H/4, W/4]
        feats1 = F.interpolate(feats1, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Build features for Side-ViT-2 ----- 
        feats2 = self.proj_sv2(f4)                    # [in_ch, H/4, W/4]
        feats2 = F.interpolate(feats2, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Side-ViT predictions -----
        vit_out1 = self.sidevit1(feats1, K_value, Q_value)
        vit_out2 = self.sidevit2(feats2, K_value, Q_value)
        vit_out3 = self.side_vit_cnn(x, K_value, Q_value)

        # Combine and classify
        combined = torch.cat([vit_out1, vit_out2, vit_out3], dim=1)  # [batch, 4]
        logits = self.fc(combined)
        return logits


class ResNetSideViTClassifier_2(nn.Module):
    def __init__(
        self,
        side_vit1: FineGrainedPromptTuning,
        side_vit2: FineGrainedPromptTuning,
        side_vit_cnn: FineGrainedPromptTuning,
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
        self.proj_sv1 = nn.Conv2d(c2 + c3, in_ch, kernel_size=1)
        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1)

        # Side-ViT classifiers
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        # MLP head with dropout for regularization
        hidden_dim = getattr(cfg, 'mlp_hidden_dim', 12)
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
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
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        feats23 = torch.cat([f2, f3_up], dim=1)
        feats1 = self.proj_sv1(feats23)                 # [in_ch, H/4, W/4]
        feats1 = F.interpolate(feats1, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Build features for Side-ViT-2 ----- 
        feats2 = self.proj_sv2(f4)                    # [in_ch, H/4, W/4]
        feats2 = F.interpolate(feats2, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Side-ViT predictions -----
        vit_out1 = self.sidevit1(feats1, K_value, Q_value)
        vit_out2 = self.sidevit2(feats2, K_value, Q_value)
        vit_out3 = self.side_vit_cnn(x, K_value, Q_value)

        # Combine and classify
        combined = torch.cat([vit_out1, vit_out2, vit_out3], dim=1)  # [batch, 4]
        logits = self.mlp(combined)
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ResnetSideViTClassifier_3(nn.Module):
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

        # Feature Pyramid: lateral convs with normalization, activation, dropout
        C = cfg.dataset.image_channel_num
        self.lateral2 = nn.Sequential(
            nn.Conv2d(channels[1], C, kernel_size=1, bias=False),
            nn.GroupNorm(8, C),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.lateral3 = nn.Sequential(
            nn.Conv2d(channels[2], C, kernel_size=1, bias=False),
            nn.GroupNorm(8, C),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.lateral4 = nn.Sequential(
            nn.Conv2d(channels[3], C, kernel_size=1, bias=False),
            nn.GroupNorm(8, C),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        # Side-ViT backbones (black-box)
        self.sidevit1    = side_vit1
        self.sidevit2    = side_vit2
        self.sidevit_cnn = side_vit_cnn

        # Stochastic depth
        # Fusion normalization and dropout before classifier (keep head unchanged)
        total_dim = 2 * 3
        self.fusion_norm    = nn.LayerNorm(total_dim)

        # Classification head (unchanged)
        hidden = getattr(cfg, 'mlp_hidden_dim', 16)
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout if hasattr(cfg, 'dropout') else 0.2),
            nn.Linear(hidden, cfg.dataset.num_classes)
        )

        # Initialize new layers
        for m in [*self.lateral2, *self.lateral3, *self.lateral4, self.fusion_norm, *self.classifier]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, K_value, Q_value) -> torch.Tensor:
        # Backbone forward
        x0 = self.stem(x)
        f1 = self.layer1(x0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # FPN with dropout-wrapped laterals
        p4 = self.lateral4(f4)
        p3 = self.lateral3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode='nearest')
        p2 = self.lateral2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode='nearest')

        # Resize for ViT inputs
        feat1 = F.interpolate(p2, size=(128,128), mode='bilinear', align_corners=False)
        feat2 = F.interpolate(p3, size=(128,128), mode='bilinear', align_corners=False)
        feat3 = x

        # Stochastic input dropouts
        feat1 = F.dropout(feat1, p=0.1, training=self.training)
        feat2 = F.dropout(feat2, p=0.1, training=self.training)
        feat3 = F.dropout(feat3, p=0.1, training=self.training)

        # Side-ViT forward
        out1 = self.sidevit1(feat1, K_value, Q_value)
        out2 = self.sidevit2(feat2, K_value, Q_value)
        out3 = self.sidevit_cnn(feat3, K_value, Q_value)

        # Combine and apply stochastic depth
        combined = torch.cat([out1, out2, out3], dim=1)

        combined = self.drop_path(combined)
        fused = self.fusion_norm(combined)
        
        logits = self.classifier(combined)
        return logits
