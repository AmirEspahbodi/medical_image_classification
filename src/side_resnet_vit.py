import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .bridge import FineGrainedPromptTuning
from typing import Any

class ResNetSideViTClassifier_FFN_FC(nn.Module):
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
        self.fc = nn.Linear(4, cfg.dataset.num_classes)

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
        feats1 = self.encdec1(feats1)                      # robust blending
        feats1 = F.interpolate(feats2, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Build features for Side-ViT-2 ----- 
        feats2 = self.proj_sv2(f4)                    # [in_ch, H/4, W/4]
        feats2 = self.encdec2(feats2)                      # robust blending
        feats2 = F.interpolate(feats1, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Side-ViT predictions -----
        vit_out1 = self.sidevit1(feats1, K_value, Q_value)
        vit_out2 = self.sidevit2(feats2, K_value, Q_value)

        # Combine and classify
        combined = torch.cat([vit_out1, vit_out2], dim=1)  # [batch, 4]
        logits = self.fc(combined)
        return logits


class ResNetSideViTClassifier_FFN_MLP(nn.Module):
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
        feats23 = torch.cat([f2, f3_up], dim=1)
        feats1 = self.proj_sv1(feats23)                 # [in_ch, H/4, W/4]
        feats1 = self.encdec1(feats1)                      # robust blending
        feats1 = F.interpolate(feats1, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Build features for Side-ViT-2 ----- 
        feats2 = self.proj_sv2(f4)                    # [in_ch, H/4, W/4]
        feats2 = self.encdec2(feats2)                      # robust blending
        feats2 = F.interpolate(feats2, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Side-ViT predictions -----
        vit_out1 = self.sidevit1(feats1, K_value, Q_value)
        vit_out2 = self.sidevit2(feats2, K_value, Q_value)

        # Combine and classify
        combined = torch.cat([vit_out1, vit_out2], dim=1)  # [batch, 4]
        logits = self.mlp(combined)
        return logits

class ResNetSideViTClassifier_FC(nn.Module):
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
        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1)

        # Side-ViT classifiers
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2

        # MLP head with dropout for regularization
        hidden_dim = getattr(cfg, 'mlp_hidden_dim', 8)
        self.fc = nn.Linear(4, cfg.dataset.num_classes)

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

        # Combine and classify
        combined = torch.cat([vit_out1, vit_out2], dim=1)  # [batch, 4]
        logits = self.fc(combined)
        return logits


class ResNetSideViTClassifier_MLP(nn.Module):
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
        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1)

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
        feats23 = torch.cat([f2, f3_up], dim=1)
        feats1 = self.proj_sv1(feats23)                 # [in_ch, H/4, W/4]
        feats1 = F.interpolate(feats1, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Build features for Side-ViT-2 ----- 
        feats2 = self.proj_sv2(f4)                    # [in_ch, H/4, W/4]
        feats2 = F.interpolate(feats2, size=(128, 128), mode='bilinear', align_corners=False)

        # ----- Side-ViT predictions -----
        vit_out1 = self.sidevit1(feats1, K_value, Q_value)
        vit_out2 = self.sidevit2(feats2, K_value, Q_value)

        # Combine and classify
        combined = torch.cat([vit_out1, vit_out2], dim=1)  # [batch, 4]
        logits = self.mlp(combined)
        return logits

class ResNetSideViTClassifier_SV(nn.Module):
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
        self.proj_sv1 = nn.Conv2d(c2+c3, in_ch, kernel_size=1)
        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1)

        # Side-ViT classifiers
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2

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
        probs1 = self.sidevit1(feats1, K_value, Q_value)
        probs2 = self.sidevit2(feats2, K_value, Q_value)

        # soft-voting
        probs = (probs1 + probs2) / 2
        return probs


### - - - - - - - - - - - - - - - - - - - - - -
 
class ResNetSideViTClassifier_FC_CNNVIT(nn.Module):
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


class ResNetSideViTClassifier_MLP_CNNVIT_old(nn.Module):
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

class ResNetSideViTClassifier_SV_CNNVIT(nn.Module):
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
        self.proj_sv1 = nn.Conv2d(c2+c3, in_ch, kernel_size=1)
        self.proj_sv2 = nn.Conv2d(c4, in_ch, kernel_size=1)

        # Side-ViT classifiers
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn


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
        probs1 = self.sidevit1(feats1, K_value, Q_value)
        probs2 = self.sidevit2(feats2, K_value, Q_value)
        probs3 = self.side_vit_cnn(x, K_value, Q_value)

        # soft-voting
        probs = (probs1 + probs2 + probs3) / 3
        return probs


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Squeeze-and-Excitation block tailored for small total_dim
def se_block(channels: int, reduction: int = 2) -> nn.Module:
    inner_dim = max(channels // reduction, 1)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(channels, inner_dim, kernel_size=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_dim, channels, kernel_size=1, bias=False),
        nn.Sigmoid()
    )

class ResNetSideViTClassifier_MLP_CNNVIT(nn.Module):
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
        attn     = self.se(combined)
        fused    = combined * attn

        x = self.norm(fused)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


