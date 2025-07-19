import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .bridge import FineGrainedPromptTuning
from typing import Any


class ResNetSideViTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vit_embed_dim: int,
        side_vit: FineGrainedPromptTuning,
        cfg: Any,
        resnet_variant: str = 'resnet18',
        pretrained: bool = True,
    ):
        super().__init__()
        # Load ResNet backbone up to layer3
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

        # Keep initial layers
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # (B, c1, H/4, W/4)
        self.layer2 = backbone.layer2  # (B, c2, H/8, W/8)
        self.layer3 = backbone.layer3  # (B, c3, H/16, W/16)

        # Projection: reduce channels to Side-ViT's expected input channels
        # We'll use 1x1 conv as "FC" per spatial location
        self.proj_conv = nn.Conv2d(c2 + c3, cfg.dataset.image_channel_num, kernel_size=1)

        # Side-ViT module
        self.sidevit = side_vit

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(vit_embed_dim),
            nn.Linear(vit_embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, K_value, Q_value) -> torch.Tensor:
        # Backbone feature extraction
        x = self.stem(x)        # (B, 64, 32, 32)
        x = self.layer1(x)      # (B, c1, 32, 32)
        f2 = self.layer2(x)     # (B, c2, 16, 16)
        f3 = self.layer3(f2)    # (B, c3, 8, 8)

        # Upsample f3 to f2's spatial dims
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)

        # Concatenate
        feats = torch.cat([f2, f3_up], dim=1)  # (B, c2+c3, 16, 16)

        # Project channels
        feats = self.proj_conv(feats)         # (B, in_ch, 16, 16)

        # Upsample to 224x224 for Side-ViT
        feats = F.interpolate(feats, size=(224, 224), mode='bilinear', align_corners=False)

        # Side-ViT forward
        vit_out = self.sidevit(feats, K_value, Q_value)         # (B, vit_embed_dim)

        # Classification head
        logits = self.classifier(vit_out)     # (B, num_classes)
        probs = F.softmax(logits, dim=-1)
        return probs
