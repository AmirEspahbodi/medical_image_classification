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
        self.proj_sv2 = nn.Conv2d(c3 + c4, in_ch, kernel_size=1)

        # Encoder-Decoder feed-forward modules for robust feature blending
        
        mlp_dropout = getattr(cfg, 'mlp_dropout', 0.3)
        
        hidden_ff = in_ch * 2
        self.encdec1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ff, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Conv2d(hidden_ff, in_ch, kernel_size=1),
        )
        self.encdec2 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ff, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Conv2d(hidden_ff, in_ch, kernel_size=1),
        )

        # Side-ViT classifiers
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2

        # MLP head with dropout for regularization
        hidden_dim = getattr(cfg, 'mlp_hidden_dim', 8)
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
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
        feats12 = torch.cat([f2, f3_up], dim=1)            # [c1+c2, H/4, W/4]
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
