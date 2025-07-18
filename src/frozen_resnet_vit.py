import torch
import torch.nn as nn
import torchvision.models as models
from .builder import build_frozen_encoder
from transformers import ViTConfig


# class CombinedResNetViT(nn.Module):
#     def __init__(
#         self,
#         cfg,
#         resnet_variant: str,
#         num_classes: int,
#         # token_imp: str,
#         # token_ratio: float,
#         # layers_to_extract: str,
#     ):
#         super().__init__()
#         # 1. Backbone CNN
#         assert resnet_variant in ['resnet18', 'resnet50', 'resnet101'], "Unsupported ResNet variant"
#         self.resnet = getattr(models, resnet_variant)(pretrained=True)

#         # 2. Frozen ViT encoder
#         frozen_vit, frozen_config = build_frozen_encoder(
#             cfg
#         )
#         frozen_vit = frozen_vit.to(cfg.base.device)
        
#         # Extract embeddings, encoder, norm, pooler
#         self.vit_embeddings = frozen_vit.vit.embeddings
#         self.vit_encoder = frozen_vit.vit.encoder
#         self.vit_layernorm = frozen_vit.vit.layernorm
#         self.vit_pooler = frozen_vit.vit.pooler

#         # 3. Projection of CNN features to ViT hidden size
#         hidden_size = frozen_config.hidden_size
#         c2 = self.resnet.layer2[-1].conv3.out_channels if hasattr(self.resnet.layer2[-1], 'conv3') else self.resnet.layer2[-1].conv2.out_channels
#         c3 = self.resnet.layer3[-1].conv3.out_channels if hasattr(self.resnet.layer3[-1], 'conv3') else self.resnet.layer3[-1].conv2.out_channels
#         self.proj2 = nn.Linear(c2, hidden_size)
#         self.proj3 = nn.Linear(c3, hidden_size)

#         # 4. Classification head
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, num_classes),
#             nn.Softmax(dim=-1)
#         )

#     def forward(self, x: torch.Tensor, interpolate_pos_encoding=True) -> torch.Tensor:
#         # Ensure model parameters are on the same device as input
#         model_device = next(self.parameters()).device
#         if x.device != model_device:
#             self.to(x.device)

#         # 1. CNN feature extraction up to layer3
#         feat = self.resnet.conv1(x)
#         feat = self.resnet.bn1(feat)
#         feat = self.resnet.relu(feat)
#         feat = self.resnet.maxpool(feat)
#         feat = self.resnet.layer1(feat)
#         out2 = self.resnet.layer2(feat)  # [B, C2, H2, W2]
#         out3 = self.resnet.layer3(out2)  # [B, C3, H3, W3]

#         # 2. Flatten spatial dims → tokens
#         B, C2, H2, W2 = out2.shape
#         tokens2 = out2.flatten(2).permute(0, 2, 1)  # [B, N2, C2]
#         B, C3, H3, W3 = out3.shape
#         tokens3 = out3.flatten(2).permute(0, 2, 1)  # [B, N3, C3]

#         # Project to hidden size
#         tokens2 = self.proj2(tokens2)  # [B, N2, hidden]
#         tokens3 = self.proj3(tokens3)  # [B, N3, hidden]

#         # 3. Prepare embeddings: concat tokens with CLS & positional embeddings
#         tokens = torch.cat([tokens2, tokens3], dim=1)  # [B, N, hidden]
#         B = x.size(0)
#         # prepend CLS token
#         cls_tokens = self.vit_embeddings.cls_token.expand(B, -1, -1)
#         embeddings = torch.cat([cls_tokens, tokens], dim=1)  # [B, 1+N, hidden]
#         # add only CLS positional embedding (others skipped due to mismatch)
#         pos_cls = self.vit_embeddings.position_embeddings[:, :1, :]
#         embeddings[:, :1, :] = embeddings[:, :1, :] + pos_cls
#         embeddings = self.vit_embeddings.dropout(embeddings)

#         # 4. Transformer encoding
#         encoder_outputs = self.vit_encoder(
#             embeddings,
#             output_attentions=False,
#             output_hidden_states=False,
#             return_dict=False,
#             # interpolate_pos_encoding=True
#         )
#         seq_out = encoder_outputs[0]  # [B, N+1, hidden]
#         seq_out = self.vit_layernorm(seq_out)
#         pooled = self.vit_pooler(seq_out)  # [B, hidden]

#         # 5. Classification
#         logits = self.classifier(pooled)
#         return logits



class CombinedResNetViT(nn.Module):
    def __init__(
        self,
        cfg,
        resnet_variant: str,
        num_classes: int,
        pool_size2: int = 14,
        pool_size3: int = 7,
    ):
        super().__init__()
        # 1. Backbone CNN
        assert resnet_variant in ['resnet18', 'resnet50', 'resnet101'], "Unsupported ResNet variant"
        self.resnet = getattr(models, resnet_variant)(pretrained=True)

        # Pooling to reduce tokens
        self.pool2 = nn.AdaptiveAvgPool2d((pool_size2, pool_size2))
        self.pool3 = nn.AdaptiveAvgPool2d((pool_size3, pool_size3))

        # 2. Frozen ViT encoder
        frozen_vit, frozen_config = build_frozen_encoder(
            cfg
        )
        frozen_vit = frozen_vit.to(cfg.base.device)
        
        # Extract embeddings, encoder, norm, pooler
        self.vit_embeddings = frozen_vit.vit.embeddings
        self.vit_encoder = frozen_vit.vit.encoder
        self.vit_layernorm = frozen_vit.vit.layernorm
        self.vit_pooler = frozen_vit.vit.pooler
        
        # Extract embedding & encoder components
        self.vit_embeddings = frozen_vit.vit.embeddings
        self.vit_encoder = frozen_vit.vit.encoder
        self.vit_layernorm = frozen_vit.vit.layernorm
        self.vit_pooler = frozen_vit.vit.pooler

        # 3. Projection of CNN features to ViT hidden size
        hidden_size = frozen_config.hidden_size
        c2 = self.resnet.layer2[-1].conv3.out_channels if hasattr(self.resnet.layer2[-1], 'conv3') else self.resnet.layer2[-1].conv2.out_channels
        c3 = self.resnet.layer3[-1].conv3.out_channels if hasattr(self.resnet.layer3[-1], 'conv3') else self.resnet.layer3[-1].conv2.out_channels
        self.proj2 = nn.Linear(c2, hidden_size)
        self.proj3 = nn.Linear(c3, hidden_size)

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor, interpolate_pos_encoding=True) -> torch.Tensor:
        # Ensure same device
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
        cls_tokens = self.vit_embeddings.cls_token.expand(B, -1, -1)
        embeddings = torch.cat([cls_tokens, tokens], dim=1)
        # Add only CLS positional embedding to avoid mismatches
        embeddings[:, :1, :] += self.vit_embeddings.position_embeddings[:, :1, :]
        embeddings = self.vit_embeddings.dropout(embeddings)

        # 4. Transformer encoding
        encoder_outputs = self.vit_encoder(
            embeddings,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            # interpolate_pos_encoding=True
        )
        seq_out = encoder_outputs[0]
        seq_out = self.vit_layernorm(seq_out)
        pooled = self.vit_pooler(seq_out)

        # 5. Classification
        logits = self.classifier(pooled)
        return logits