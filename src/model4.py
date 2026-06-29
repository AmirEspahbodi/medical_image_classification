import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Any
from typing import List


class MultiScaleCoAtNetBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_chans: int = 3,
        pretrained: bool = True,
        backbone_trainable_layers: List[int] = [],
    ):
        super().__init__()

        self.cnn_backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        trainable_block_names = {f"blocks.{i}" for i in backbone_trainable_layers}

        trainable_params_count = 0
        total_params_count = 0

        if backbone_trainable_layers == [0, 1, 2, 3] or backbone_trainable_layers == (
            0,
            1,
            2,
            3,
        ):
            for name, param in self.cnn_backbone.named_parameters():
                total_params_count += param.numel()
                if any(block_name in name for block_name in trainable_block_names):
                    trainable_params_count += param.numel()
        else:
            for name, param in self.cnn_backbone.named_parameters():
                total_params_count += param.numel()
                # Check if the parameter belongs to one of the specified trainable blocks
                if any(block_name in name for block_name in trainable_block_names):
                    param.requires_grad = True
                    trainable_params_count += param.numel()
                else:
                    param.requires_grad = False

        print(
            f"--- Initialized CNN Backbone: {model_name} (pretrained={pretrained}) ---"
        )
        feature_info = self.cnn_backbone.feature_info.channels()
        self.channels = feature_info
        print(f"    Feature map channels extracted: {feature_info}")

        if not backbone_trainable_layers:
            print("    All backbone layers are FROZEN.")
        else:
            frozen_params_count = total_params_count - trainable_params_count
            print(f"    Trainable blocks: {backbone_trainable_layers}")
            print(f"    Trainable parameters: {trainable_params_count:,}")
            print(f"    Frozen parameters: {frozen_params_count:,}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feature_maps = self.cnn_backbone(x)
        return feature_maps


class GatedAttentionModule(nn.Module):
    def __init__(
        self, low_level_channels: int, high_level_channels: int, output_channels: int
    ):
        super().__init__()
        # Convolution to generate attention map from high-level features
        self.attn_conv = nn.Conv2d(
            high_level_channels, low_level_channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

        # 1x1 convolution to process the attended features
        self.proj_conv = nn.Conv2d(
            low_level_channels, output_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, low_level_feat: torch.Tensor, high_level_feat: torch.Tensor
    ) -> torch.Tensor:
        # Upsample high-level features to match the spatial dimensions of low-level features
        high_level_upsampled = F.interpolate(
            high_level_feat,
            size=low_level_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Generate spatial attention map
        attention_map = self.attn_conv(high_level_upsampled)
        attention_map = self.sigmoid(attention_map)

        # Apply attention to low-level features
        attended_feat = low_level_feat * attention_map

        # Project and normalize the result
        output = self.proj_conv(attended_feat)
        output = self.bn(output)
        output = self.relu(output)

        return output


class SpatialCrossAttention(nn.Module):
    """
    Fuses a processed feature map (query) with a raw image (context) using spatial
    cross-attention. This allows the model to use semantic context to select relevant
    details from the raw image before feeding the result to a Side-ViT.
    """

    def __init__(
        self, query_channels: int, context_channels: int, output_channels: int
    ):
        super().__init__()
        inter_channels = query_channels // 2 if query_channels > 1 else 1
        self.query_conv = nn.Conv2d(query_channels, inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(context_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(context_channels, query_channels, kernel_size=1)
        self.proj_conv = nn.Conv2d(query_channels, output_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, output_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, query_feat: torch.Tensor, context_feat: torch.Tensor
    ) -> torch.Tensor:
        B, C_q, H, W = query_feat.shape

        # [FIX] Resize context to match query's spatial dimensions internally
        context_feat_resized = F.interpolate(
            context_feat, size=(H, W), mode="bilinear", align_corners=False
        )

        # Generate Q, K, V
        q = (
            self.query_conv(query_feat).view(B, -1, H * W).permute(0, 2, 1)
        )  # (B, H*W, C_inter)
        k = self.key_conv(context_feat_resized).view(B, -1, H * W)  # (B, C_inter, H*W)
        v = self.value_conv(context_feat_resized).view(B, -1, H * W)  # (B, C_q, H*W)

        # Calculate attention scores
        # q: (B, H*W, C_inter), k: (B, C_inter, H*W) -> scores: (B, H*W, H*W)
        attention_scores = torch.bmm(q, k)
        attention_probs = self.softmax(attention_scores)

        # Apply attention to Value
        # v: (B, C_q, H*W), probs.T: (B, H*W, H*W) -> attended: (B, C_q, H*W)
        attended_v = torch.bmm(v, attention_probs.permute(0, 2, 1))
        attended_v = attended_v.view(B, C_q, H, W)

        # Add residual connection and project
        fused_feat = self.proj_conv(query_feat + attended_v)

        return self.norm(fused_feat)


class CoAtNetSideViTClassifier_4(nn.Module):
    def __init__(self, side_vit1: nn.Module, side_vit2: nn.Module, cfg: Any):
        super().__init__()
        print("CoAtNetSideViTClassifier_4")
        backbone_trainable_layers = [
            int(i) - 1 for i in cfg.network.backbone_trainable_layers
        ]
        self.vit1_feature_strame = [int(i) - 1 for i in cfg.network.vit1_feature_strame]
        self.vit2_feature_strame = [int(i) - 1 for i in cfg.network.vit2_feature_strame]

        self.cfg = cfg
        self.num_classes = cfg.dataset.num_classes

        self.cnn_backbone = MultiScaleCoAtNetBackbone(
            model_name="coatnet_0_rw_224",
            pretrained=True,
            in_chans=cfg.dataset.image_channel_num,
            backbone_trainable_layers=backbone_trainable_layers,
        )

        feat_dims = self.cnn_backbone.channels

        NUM_VIT_BRANCHS = 2
        proj_channels = 64

        branch1_dim = [
            feat_dims[i] for i in self.vit1_feature_strame
        ]  # for example, layer 2, 3 :=  192 + 384 = 576
        branch2_dim = [feat_dims[i] for i in self.vit2_feature_strame]

        # --- Feature Preparation Paths ---
        if len(self.vit1_feature_strame) == 2:
            self.gate1 = GatedAttentionModule(*branch1_dim, 64)
        else:
            self.proj_sv1 = nn.Conv2d(
                sum(branch1_dim), proj_channels, kernel_size=1, bias=False
            )

        if len(self.vit2_feature_strame) == 2:
            self.gate2 = GatedAttentionModule(*branch2_dim, 64)
        else:
            self.proj_sv2 = nn.Conv2d(
                sum(branch2_dim), proj_channels, kernel_size=1, bias=False
            )

        # --- Spatial Cross-Attention Fusion for ViT Inputs ---
        self.spatial_fusion1 = SpatialCrossAttention(
            64, cfg.dataset.image_channel_num, cfg.dataset.image_channel_num
        )
        self.spatial_fusion2 = SpatialCrossAttention(
            64, cfg.dataset.image_channel_num, cfg.dataset.image_channel_num
        )

        # --- Side-ViT Modules ---
        self.side_vit1 = side_vit1
        self.side_vit2 = side_vit2

        # --- Fusion of Side-ViT Outputs ---
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(self.num_classes * NUM_VIT_BRANCHS),
            nn.Linear(
                self.num_classes * NUM_VIT_BRANCHS,
                self.num_classes * NUM_VIT_BRANCHS * 2,
            ),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_classes * NUM_VIT_BRANCHS * 2, self.num_classes),
        )

    def forward(self, x: torch.Tensor, key_states, value_states) -> torch.Tensor:
        features = self.cnn_backbone(x)

        if len(self.vit1_feature_strame) == 2:
            proc_feat1 = self.gate1(*[features[f] for f in self.vit1_feature_strame])
        else:
            proc_feat1 = self.proj_sv1(features[self.vit1_feature_strame[0]])

        if len(self.vit2_feature_strame) == 2:
            proc_feat2 = self.gate2(*[features[f] for f in self.vit2_feature_strame])
        else:
            proc_feat2 = self.proj_sv2(features[self.vit2_feature_strame[0]])

        # [FIX] Pass raw image 'x' directly. Resizing is now handled inside SpatialCrossAttention.
        vit_input1 = self.spatial_fusion1(proc_feat1, x)
        vit_input2 = self.spatial_fusion2(proc_feat2, x)

        vit_input1 = F.interpolate(
            vit_input1,
            size=(self.cfg.network.side_input_size, self.cfg.network.side_input_size),
            mode="bilinear",
            align_corners=False,
        )
        vit_input2 = F.interpolate(
            vit_input2,
            size=(self.cfg.network.side_input_size, self.cfg.network.side_input_size),
            mode="bilinear",
            align_corners=False,
        )

        vit_out1 = self.side_vit1(vit_input1, key_states, value_states)
        vit_out2 = self.side_vit2(vit_input2, key_states, value_states)

        features = torch.cat([vit_out1, vit_out2], dim=1)
        logits = self.classifier_head(features)
        return logits
