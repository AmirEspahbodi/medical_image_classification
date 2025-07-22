import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from typing import Any, Tuple


class ResNetSideViTClassifier_MLP_CNNVIT2(nn.Module):
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
        # attn     = self.se(combined)
        # fused    = combined * attn

        # x = self.norm(fused)
        # x = self.dropout(x)
        logits = self.classifier(combined) #combined
        return logits

#########################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict
from typing import Any, Dict, List
import timm

# --- Component 1: Modular Backbone Feature Extractor ---
class CoAtNetFeatureExtractor(nn.Module):
    """
    A wrapper for the CoAtNet backbone to extract intermediate features from its stages.
    This encapsulates the backbone logic and provides a clean interface.
    """
    def __init__(self, coatnet_variant: str = 'coatnet_0.untrained', pretrained: bool = True):
        super().__init__()
        # Load the specified CoAtNet model from the timm library.
        # Timm provides access to many SOTA models with pretrained weights.
        # Note: As of late 2023, official 'coatnet_0' weights might not be in timm's default set.
        # You might need to load them manually or use a similar variant.
        # For demonstration, we use 'coatnet_0.untrained' to show architecture.
        # In a real scenario, you would use pretrained=True with a valid model name.
        self.backbone = timm.create_model(coatnet_variant, pretrained=pretrained, features_only=True)
        
        # The channel dimensions for coatnet_0 outputs
        # From timm's documentation or by inspecting the model:
        # s0: 64, s1: 64, s2: 96, s3: 192, s4: 384
        self.out_channels = self.backbone.feature_info.channels()
        print(f"CoAtNet backbone loaded. Feature channels: {self.out_channels}")


    def forward(self, x: torch.Tensor) -> Dict:
        # The `features_only=True` flag makes the model return a list of feature maps.
        features = self.backbone(x)
        # Return as a named dictionary for clarity, matching FPN's expected input format.
        # We map the list indices to stage names.
        return {f's{i+1}': feat for i, feat in enumerate(features[1:])} # Skip s0 (stem) for FPN

# --- Component 2: Convolutional Stem for Raw Image Path ---
class ConvStem(nn.Module):
    """
    A convolutional stem to replace the standard ViT patchify layer.
    This provides more stable training and better performance by introducing
    convolutional inductive bias early on.
    Ref: "Early Convolutions Help Transformers See Better" (Xiao et al., 2021)
    """
    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        super().__init__()
        # A simple stem with two convolutional layers.
        # This progressively downsamples the image and creates patch embeddings.
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

# --- Component 3: Feature Pyramid Network for Multi-Scale Fusion ---
class FPN_Neck(nn.Module):
    """
    A Feature Pyramid Network (FPN) neck to fuse multi-scale features
    from the CoAtNet backbone. This enriches shallow-layer features with
    deep-layer semantics.
    """
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )
        # FPN returns an OrderedDict, we want to return a list of tensors
        # in the order of the pyramid levels (from finest to coarsest).
        self.out_channels = out_channels

    def forward(self, x: Dict) -> Dict:
        return self.fpn(x)

# --- Component 4: Attention-based Fusion for ViT Outputs ---
class AttentionFusion(nn.Module):
    """
    Intelligently fuses the outputs of the three side-ViTs using a simple
    self-attention mechanism. This allows the model to dynamically weigh the
    importance of each ViT's contribution for a given sample.
    """
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_vits = 3
        self.feature_dim = feature_dim
        
        # We treat the 3 ViT outputs as a sequence of length 3.
        # A standard multi-head attention module is used for fusion.
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # A layer norm for stability
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, vit_outputs: List) -> torch.Tensor:
        # vit_outputs: A list of 3 tensors, each of shape [batch_size, feature_dim]
        
        # Stack the outputs to form a sequence: [batch_size, seq_len=3, feature_dim]
        x = torch.stack(vit_outputs, dim=1)
        
        # The query, key, and value are all the same sequence (self-attention).
        # We are interested in the aggregated output, so we take the first element of the tuple.
        attn_output, _ = self.attention(x, x, x)
        
        # Add a residual connection and normalize
        x = self.norm(x + attn_output)
        
        # Flatten the fused features to be passed to the MLP
        # [batch_size, 3 * feature_dim]
        return x.flatten(1)

# --- The Final Integrated Model ---
class ResNetSideViTClassifier_MLP_CNNVIT(nn.Module):
    def __init__(
        self,
        side_vit1,
        side_vit2,
        side_vit_cnn,
        cfg: Any,
        backbone_variant: str = 'coatnet_0.untrained',
        pretrained: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        num_classes: int = 2

        # 1. Instantiate the CoAtNet Feature Extractor Backbone
        self.backbone = CoAtNetFeatureExtractor(backbone_variant, pretrained_backbone)
        # [s1, s2, s3, s4] -> channels  for coatnet_0
        backbone_channels = self.backbone.out_channels
        
        # 2. Instantiate the FPN Neck
        # We will feed features from s1, s2, and s4 to the FPN, as per our design.
        # The FPN will output features with a consistent channel dimension.
        fpn_out_channels = 256
        self.fpn_neck = FPN_Neck(
            in_channels_list=[backbone_channels, backbone_channels, backbone_channels],
            out_channels=fpn_out_channels
        )

        # 3. Instantiate the Convolutional Stem for the raw image ViT
        # The out_channels should match the expected in_channels of side_vit_cnn
        # We assume side_vit_cnn has an `in_channels` attribute for this.
        vit_cnn_in_channels = 128 # Example value
        self.conv_stem = ConvStem(
            in_channels=cfg.dataset.image_channel_num,
            out_channels=vit_cnn_in_channels,
            patch_size=4 # The stem effectively creates patches of size 4x4
        )

        # 4. Store the black-box Side-ViT classifiers
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn
        
        # 5. Instantiate the Attention Fusion module and the final MLP head
        # We assume the black-box ViTs output features of a certain dimension before the final layer.
        # For this demo, we assume they all output `num_classes` logits.
        # A better approach is to modify the ViTs to return pre-logit features.
        # Let's assume the feature dimension is `num_classes` for simplicity.
        vit_feature_dim = num_classes
        self.attention_fusion = AttentionFusion(feature_dim=vit_feature_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(self.num_vits * vit_feature_dim, 16),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )

    def forward(self, x: torch.Tensor, K_value: Any, Q_value: Any) -> torch.Tensor:
        # Phase 1: Backbone Feature Extraction
        # Get multi-scale features from the CoAtNet backbone.
        backbone_feats = self.backbone(x)
        
        # Phase 2: FPN Fusion
        # Select the features for the FPN and process them.
        fpn_input = OrderedDict()
        fpn_input['s1'] = backbone_feats['s1']
        fpn_input['s2'] = backbone_feats['s2']
        fpn_input['s4'] = backbone_feats['s4']
        fpn_output = self.fpn_neck(fpn_input)

        # Phase 3: Prepare inputs for each Side-ViT
        # Input for side_vit1 from FPN level corresponding to s2
        feats_for_sv1 = fpn_output['1'] # FPN names outputs '0', '1', '2', etc.
        
        # Input for side_vit2 from FPN level corresponding to s4
        feats_for_sv2 = fpn_output['2']
        
        # Input for side_vit_cnn from our custom ConvStem
        feats_for_svc = self.conv_stem(x)
        
        # Phase 4: Get predictions from each Side-ViT
        # Note: The black-box ViTs must be adapted to accept the channel dimensions
        # from the FPN (256) and ConvStem (128). This is a critical assumption.
        vit_out1 = self.sidevit1(feats_for_sv1, K_value, Q_value)
        vit_out2 = self.sidevit2(feats_for_sv2, K_value, Q_value)
        vit_out3 = self.side_vit_cnn(feats_for_svc, K_value, Q_value)
        
        # Phase 5: Fuse outputs and classify
        # Use the attention fusion module
        fused_features = self.attention_fusion([vit_out1, vit_out2, vit_out3])
        
        # Final classification
        logits = self.mlp_head(fused_features)
        
        return logits
