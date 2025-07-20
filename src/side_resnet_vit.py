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

        # Initial layers
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # (B, c1, H/4, W/4)
        self.layer2 = backbone.layer2  # (B, c2, H/8, W/8)
        self.layer3 = backbone.layer3  # (B, c3, H/16, W/16)

        # 1x1 conv to reduce channels to Side-ViT in_chans
        in_ch = cfg.dataset.image_channel_num
        self.proj_conv = nn.Conv2d(c2 + c3, in_ch, kernel_size=1)

        # Side-ViT
        self.sidevit = side_vit

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(vit_embed_dim),
            nn.Linear(vit_embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, K_value, Q_value) -> torch.Tensor:
        # x: (B, 3, H, W)
        # Extract ResNet features
        x = self.stem(x)            # (B,64,H/4,W/4)
        x = self.layer1(x)          # (B,c1,H/4,W/4)
        f2 = self.layer2(x)         # (B,c2,H/8,W/8)
        f3 = self.layer3(f2)        # (B,c3,H/16,W/16)

        # Align spatial dims
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        feats = torch.cat([f2, f3_up], dim=1)  # (B,c2+c3,H/8,W/8)

        # Project channels -> (B,in_ch,H/8,W/8)
        feats = self.proj_conv(feats)

        feats = F.interpolate(feats, size=(128,128), mode='bilinear', align_corners=False)

        # Side-ViT returns either (B,embed_dim) or (B,N,embed_dim)
        vit_out = self.sidevit(feats, K_value, Q_value)

        # If sequence output, pick class token or global average
        # if vit_out.ndim == 3:
        #     # assume first token is [CLS]
        #     vit_out = vit_out[:, 0, :]

        # # Classifier
        # logits = self.classifier(vit_out)  # (B,num_classes)
        return vit_out
