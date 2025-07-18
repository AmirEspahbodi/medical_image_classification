import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import ViTConfig

from .side_vit import ViTForImageClassification as SideViT


def get_resnet_backbone(name: str, pretrained: bool = True):
    """
    Returns a ResNet backbone truncated before the final pooling & FC layer.
    Supported: 'resnet18', 'resnet50', 'resnet101'.
    """
    if name not in ['resnet18', 'resnet50', 'resnet101']:
        raise ValueError(f"Unsupported backbone: {name}")
    backbone = getattr(models, name)(pretrained=pretrained)
    # Keep layers up to layer4
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
    Combines a ResNet (blocks 1-4) backbone with a Side-ViT.
    Extracts feature maps from ResNet blocks 2 and 3,
    projects them to ViT hidden size, and injects as fine-grained
    prompts into Side-ViT. The Side-ViT's pooler output is
    fed to a final FC + softmax for classification.
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
        # 1) ResNet Backbone
        self.backbone = get_resnet_backbone(resnet_name, pretrained)

        # 2) Side-ViT Configuration
        base_cfg = ViTConfig.from_pretrained(side_pretrained_path)
        hidden_size = base_cfg.hidden_size
        side_dim = hidden_size // side_reduction_ratio
        prompt_dim = hidden_size // prompt_reduction_ratio

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
        self.side_vit = SideViT(side_cfg).to(device)

        # 3) Project ResNet features to Side-ViT hidden size
        # Determine channel dims dynamically for block2 & block3
        block2 = self.backbone[2][-1]
        block3 = self.backbone[3][-1]
        c2 = block2.conv3.out_channels if hasattr(block2, 'conv3') else block2.conv2.out_channels
        c3 = block3.conv3.out_channels if hasattr(block3, 'conv3') else block3.conv2.out_channels
        self.proj2 = nn.Conv2d(c2, hidden_size, kernel_size=1)
        self.proj3 = nn.Conv2d(c3, hidden_size, kernel_size=1)

        # 4) Final classifier (after Side-ViT pooler)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x: torch.Tensor, key_states, value_states) -> torch.Tensor:
        """
        x: input images, shape (B,3,H,W).
        Returns softmax logits of shape (B, num_classes).
        """
        # Move to correct device
        x = x.to(self.device)

        # 1) Pass through ResNet blocks
        out0 = self.backbone[0](x)
        out1 = self.backbone[1](out0)
        out2 = self.backbone[2](out1)   # feature map from block2
        out3 = self.backbone[3](out2)   # feature map from block3

        # 2) Project and flatten for prompts
        # shapes: (B, hidden_size, H2, W2) -> (B, num_patches2, hidden_size)
        f2 = self.proj2(out2)
        B, C, H2, W2 = f2.shape
        f2_tokens = f2.flatten(2).transpose(1, 2)

        f3 = self.proj3(out3)
        _, _, H3, W3 = f3.shape
        f3_tokens = f3.flatten(2).transpose(1, 2)

        # 3) Side-ViT forward with fine-grained states
        vit_outputs = self.side_vit(
            pixel_values=x,
            fine_grained_states=[f2_tokens, f3_tokens]
        )
        pooled = vit_outputs.pooler_output  # (B, hidden_size)

        # 4) Final FC + softmax
        logits = self.classifier(pooled)
        return F.softmax(logits, dim=-1)
