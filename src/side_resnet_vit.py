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

        # 1) ResNet backbone
        self.backbone = get_resnet_backbone(resnet_name, pretrained)
        self.backbone = self.backbone.to(self.device)

        # 2) Side-ViT configuration & model
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
        self.side_vit = SideViT(side_cfg).to(self.device)

        # 3) Project ResNet block2 & block3 outputs to ViT hidden size
        block2 = self.backbone[2][-1]
        block3 = self.backbone[3][-1]
        c2 = block2.conv3.out_channels if hasattr(block2, 'conv3') else block2.conv2.out_channels
        c3 = block3.conv3.out_channels if hasattr(block3, 'conv3') else block3.conv2.out_channels
        self.proj2 = nn.Conv2d(c2, hidden_size, kernel_size=1).to(self.device)
        self.proj3 = nn.Conv2d(c3, hidden_size, kernel_size=1).to(self.device)

        # 4) Final classifier
        self.classifier = nn.Linear(hidden_size, num_classes).to(self.device)

    def forward(
        self,
        x: torch.Tensor,key_states, value_states) -> torch.Tensor:
        """
        x: (B,3,H,W);  f2_tokens: (B,N2,D);  f3_tokens: (B,N3,D)
        """
        # Ensure inputs on correct device and dtype
        x = x.to(self.device)

        # 1) Forward ResNet blocks (cpu->cuda safe since params are on device)
        out = x
        for idx in range(4):
            out = self.backbone[idx](out)
            if idx == 1:
                out2 = out
            elif idx == 2:
                out3 = out

        # 2) (Optional) recompute tokens if not precomputed
        # f2_tokens = self.proj2(out2).flatten(2).transpose(1,2)
        # f3_tokens = self.proj3(out3).flatten(2).transpose(1,2)

        # 3) Side-ViT forward with fine-grained states
        vit_out = self.side_vit(
            x, key_states, value_states, 
            interpolate_pos_encoding=True
        )
        pooled = vit_out.pooler_output  # (B, hidden_size)

        # 4) Final FC + softmax
        logits = self.classifier(pooled)
        return F.softmax(logits, dim=-1)
