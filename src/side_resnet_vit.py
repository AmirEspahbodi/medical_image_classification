import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import ViTConfig

from .side_vit import ViTForImageClassification as SideViT


def get_resnet_backbone(name: str, pretrained: bool = True):
    if name not in ['resnet18', 'resnet50', 'resnet101']:
        raise ValueError(f"Unsupported backbone: {name}")
    backbone = getattr(models, name)(pretrained=pretrained)
    layers = [
        nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        ),
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    ]
    return nn.ModuleList(layers)


class ResNetSideViTClassifier(nn.Module):
    """
    ResNet (blocks 1-4) + Side-ViT classifier.
    Expects fine-grained tokens from ResNet blocks 2 & 3 as inputs.
    """
    def __init__(
        self,
        resnet_name: str,
        pretrained: bool,
        side_pretrained_path: str,
        num_classes: int,
        side_reduction_ratio: int = 2,
        prompt_reduction_ratio: int = 2,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()
        self.device = device

        # ResNet backbone
        self.backbone = get_resnet_backbone(resnet_name, pretrained)

        # Side-ViT config and model
        base_cfg = ViTConfig.from_pretrained(side_pretrained_path)
        hidden_size = base_cfg.hidden_size
        side_dim = hidden_size // side_reduction_ratio

        side_cfg = ViTConfig.from_pretrained(
            side_pretrained_path,
            num_hidden_layers=base_cfg.num_hidden_layers // 2,
            hidden_size=side_dim,
            intermediate_size=side_dim * 4,
            image_size=base_cfg.image_size,
            num_labels=num_classes,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0
        )
        self.side_vit = SideViT(side_cfg)

        # Project ResNet block2 & block3 outputs to ViT hidden size
        block2 = self.backbone[2][-1]
        block3 = self.backbone[3][-1]
        c2 = block2.conv3.out_channels if hasattr(block2, 'conv3') else block2.conv2.out_channels
        c3 = block3.conv3.out_channels if hasattr(block3, 'conv3') else block3.conv2.out_channels
        self.proj2 = nn.Conv2d(c2, hidden_size, kernel_size=1)
        self.proj3 = nn.Conv2d(c3, hidden_size, kernel_size=1)

        # Final classifier
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Move all to device

    def forward(
        self,
        x: torch.Tensor,
        f2_tokens: torch.Tensor,
        f3_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (B,3,H,W);  f2_tokens: (B,N2,D);  f3_tokens: (B,N3,D)
        """
        x = x.to(self.device)
        # Pass through ResNet to maintain compatibility if needed
        out0 = self.backbone[0](x)
        out1 = self.backbone[1](out0)
        out2 = self.backbone[2](out1)
        out3 = self.backbone[3](out2)

        # Ensure tokens on correct device
        prompts = [f2_tokens.to(self.device), f3_tokens.to(self.device)]

        vit_out = self.side_vit(pixel_values=x, fine_grained_states=prompts)
        pooled = vit_out.pooler_output  # (B, hidden_size)

        logits = self.classifier(pooled)
        return F.softmax(logits, dim=-1)