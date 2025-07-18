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
        pool_size2: int = 14,
        pool_size3: int = 7,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()
        self.device = device

        # 1. Backbone CNN
        assert resnet_name in ['resnet18', 'resnet50', 'resnet101'], "Unsupported ResNet variant"
        self.resnet = getattr(models, resnet_name)(pretrained=True)

        # Pooling to reduce tokens
        self.pool2 = nn.AdaptiveAvgPool2d((pool_size2, pool_size2))
        self.pool3 = nn.AdaptiveAvgPool2d((pool_size3, pool_size3))
        
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
        # 3. Projection of CNN features to ViT hidden size
        hidden_size = side_cfg.hidden_size
        c2 = self.resnet.layer2[-1].conv3.out_channels if hasattr(self.resnet.layer2[-1], 'conv3') else self.resnet.layer2[-1].conv2.out_channels
        c3 = self.resnet.layer3[-1].conv3.out_channels if hasattr(self.resnet.layer3[-1], 'conv3') else self.resnet.layer3[-1].conv2.out_channels
        self.proj2 = nn.Linear(c2, hidden_size)
        self.proj3 = nn.Linear(c3, hidden_size)

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=-1)
        )


    def forward(
        self,
        x: torch.Tensor,key_states, value_states) -> torch.Tensor:
         # Ensure same device
        print(f"x.shape = {x.shape}")
        device = x.device
        if next(self.parameters()).device != device:
            self.to(device)

        # 1. CNN feature extraction up to layer3
        feat = self.resnet.conv1(x)
        feat = self.resnet.bn1(feat)
        feat = self.resnet.relu(feat)
        feat = self.resnet.maxpool(feat)
        feat = self.resnet.layer1(feat)
        out2 = self.resnet.layer2(feat)
        out3 = self.resnet.layer3(out2)

        # 2. Reduce token count via pooling
        out2 = self.pool2(out2)  # [B, C2, pool2, pool2]
        out3 = self.pool3(out3)  # [B, C3, pool3, pool3]

        # Flatten spatial dims -> tokens
        B, C2, H2, W2 = out2.shape
        tokens2 = out2.flatten(2).permute(0, 2, 1)
        B, C3, H3, W3 = out3.shape
        tokens3 = out3.flatten(2).permute(0, 2, 1)

        # Project to hidden size
        tokens2 = self.proj2(tokens2)
        tokens3 = self.proj3(tokens3)

        # 3. Prepare embeddings: CLS + tokens + CLS pos encoding
        tokens = torch.cat([tokens2, tokens3], dim=1)
        cls_tokens = self.side_vit.vit.embeddings.cls_token.expand(B, -1, -1)
        embeddings = torch.cat([cls_tokens, tokens], dim=1)

        # Add only CLS positional embedding to avoid mismatches
        embeddings[:, :1, :] += self.side_vit.vit.embeddings.position_embeddings[:, :1, :]
        embeddings = self.side_vit.vit.embeddings.dropout(embeddings)

        # 2) (Optional) recompute tokens if not precomputed
        # f2_tokens = self.proj2(out2).flatten(2).transpose(1,2)
        # f3_tokens = self.proj3(out3).flatten(2).transpose(1,2)

        # 3) Side-ViT forward with fine-grained states
        vit_out = self.side_vit(
            x, key_states, value_states, interpolate_pos_encoding=True
        )
        pooled = vit_out.pooler_output  # (B, hidden_size)

        # 4) Final FC + softmax
        logits = self.classifier(pooled)
        return F.softmax(logits, dim=-1)
