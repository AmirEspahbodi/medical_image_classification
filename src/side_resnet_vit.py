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
class ResNetSideViTClassifier_MLP_CNNVIT(nn.Module):
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



