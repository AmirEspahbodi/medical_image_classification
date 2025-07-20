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

class ResNetSideViTClassifier(nn.Module):
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
            c2, c3, c4 = 128, 256, 512
        elif resnet_variant == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            c2, c3, c4 = 512, 1024, 2048
        elif resnet_variant == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
            c2, c3, c4 = 512, 1024, 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")

        for param in backbone.parameters():
            param.requires_grad = False
            
        # Initial layers
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 1x1 conv to reduce channels to Side-ViT in_chans
        in_ch = cfg.dataset.image_channel_num
        self.proj_conv1 = nn.Conv2d(c2 + c3, in_ch, kernel_size=1)
        self.proj_conv2 = nn.Conv2d(c4, in_ch, kernel_size=1)

        # Side-ViT
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2


    def forward(self, x: torch.Tensor, K_value, Q_value) -> torch.Tensor:
        with torch.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            f2 = self.layer2(x)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
            feats1 = torch.cat([f2, f3_up], dim=1)

        feats1 = self.proj_conv1(feats1)
        feats1 = F.interpolate(feats1, size=(128, 128), mode='bilinear', align_corners=False)
        feats2 = self.proj_conv2(f4)
        feats2 = F.interpolate(feats2, size=(128, 128), mode='bilinear', align_corners=False)

        vit_out1 = self.sidevit1(feats1, K_value, Q_value)
        vit_out2 = self.sidevit2(feats2, K_value, Q_value)

        # probs = (vit_out1 + vit_out2) / 2
        return vit_out1
