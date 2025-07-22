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
import timm
from torchvision.ops import FeaturePyramidNetwork
from typing import Any, List

# Assume side_vit1, side_vit2, side_vit_cnn are defined elsewhere as black boxes
# from your_project import FineGrainedPromptTuning

# =====================================================================================
# Helper Module 1: CoAtNet Feature Extractor
# =====================================================================================
class CoAtNetFeatureExtractor(nn.Module):
    """
    A wrapper around a timm-based CoAtNet model to easily extract features
    from intermediate stages. CoAtNet-0 has 5 stages (0-4).
    - s0: Conv Stem
    - s1: Conv Block (MBConv)
    - s2: Conv Block (MBConv)
    - s3: Transformer Block
    - s4: Transformer Block
    """
    def __init__(self, coatnet_variant: str = 'coatnet_0', pretrained: bool = True):
        super().__init__()
        # Using features_only=True returns a list of feature maps from each stage
        self.backbone = timm.create_model(
            coatnet_variant,
            pretrained=pretrained,
            features_only=True,
        )
        # Store channel dimensions for FPN and projection layers
        self.feature_info = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)

    def freeze_backbone(self):
        """Helper function to freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_stage(self, stage_name: str):
        """Helper to unfreeze a specific stage, e.g., 's4' or 's3'."""
        # Note: timm's feature extractor wraps stages in blocks like 'blocks.0.0' etc.
        # This requires inspecting the named_parameters to get the correct prefixes.
        # A simpler approach for CoAtNet is to unfreeze layer by layer.
        # Example for CoAtNet-0 s4 (last 2 blocks are transformer blocks)
        if stage_name == 's4':
            for param in self.backbone.blocks[3].parameters():
                 param.requires_grad = True
        elif stage_name == 's3':
            for param in self.backbone.blocks[2].parameters():
                 param.requires_grad = True
        # Add other stages if needed

# =====================================================================================
# Helper Module 2: Attention-based Fusion
# =====================================================================================
class GatedAttentionFusion(nn.Module):
    """
    Learns to weigh the outputs of the three ViTs using a simple
    gating/attention mechanism.
    """
    def __init__(self, input_dim: int, num_sources: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.num_sources = num_sources
        
        # A simple network to generate attention weights (gates)
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim * num_sources, 64),
            nn.ReLU(),
            nn.Linear(64, num_sources),
            nn.Softmax(dim=1)
        )

    def forward(self, vit_out1, vit_out2, vit_out3):
        # Assuming each vit_out has shape [batch_size, input_dim]
        combined = torch.cat([vit_out1, vit_out2, vit_out3], dim=1) # [B, 3 * D]
        
        # Generate attention weights
        attention_weights = self.attention_net(combined).unsqueeze(2) # [B, 3, 1]
        
        # Reshape inputs for weighted sum
        all_vits = torch.stack([vit_out1, vit_out2, vit_out3], dim=1) # [B, 3, D]

        # Apply attention weights
        fused_output = torch.sum(all_vits * attention_weights, dim=1) # [B, D]
        return fused_output

# =====================================================================================
# Main Model: The Advanced Hybrid Classifier
# =====================================================================================
class ResNetSideViTClassifier_MLP_CNNVIT(nn.Module):
    def __init__(
        self,
        side_vit1: nn.Module, # Assumed black box
        side_vit2: nn.Module, # Assumed black box
        side_vit_cnn: nn.Module, # Assumed black box
        cfg: Any,
        backbone_variant: str = 'coatnet_0_rw_224',
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        num_classes = 2
        vit_input_channels = cfg.dataset.image_channel_num # e.g., 3 channels for RGB
        vit_output_dim = 2 # The output feature dim from each ViT, e.g., 2

        # 1. --- ADVANCED BACKBONE ---
        # Using CoAtNet for its superior hybrid architecture
        self.backbone = CoAtNetFeatureExtractor(backbone_variant, pretrained_backbone)
        s1_ch, s2_ch, s3_ch, s4_ch = self.backbone.feature_info[1:] # Skip stem channels

        # 2. --- ADVANCED FEATURE FUSION for Side-ViT-1 ---
        # Use a Feature Pyramid Network (FPN) for rich multi-scale features
        # It takes features from s1 and s2 and fuses them
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[s1_ch, s2_ch],
            out_channels=256, # A common choice for FPN output channels
        )
        # Projection from FPN output to the side-ViT input channels
        self.proj_sv1 = nn.Conv2d(256, vit_input_channels, kernel_size=1)

        # 3. --- PROJECTION for Side-ViT-2 ---
        # Simple projection from the final transformer stage (s4)
        self.proj_sv2 = nn.Conv2d(s4_ch, vit_input_channels, kernel_size=1)

        # 4. --- SIDE-VIT CLASSIFIERS (black boxes) ---
        self.sidevit1 = side_vit1
        self.sidevit2 = side_vit2
        self.side_vit_cnn = side_vit_cnn

        # 5. --- ADVANCED FUSION of ViT OUTPUTS ---
        self.fusion_module = GatedAttentionFusion(input_dim=vit_output_dim, num_sources=3)

        # 6. --- ROBUST CLASSIFIER HEAD ---
        # A deeper MLP with BatchNorm and Dropout for regularization
        mlp_hidden_dim = 16
        dropout_rate = 0.3
        self.classifier_head = nn.Sequential(
            nn.Linear(vit_output_dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden_dim, num_classes)
        )

        # --- Initialize weights for newly added layers ---
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for FPN, projections, and classifier head."""
        for m in [self.fpn, self.proj_sv1, self.proj_sv2, self.classifier_head]:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, K_value=None, Q_value=None) -> torch.Tensor:
        # 1. --- EXTRACT HIERARCHICAL FEATURES from CoAtNet ---
        # features is a list of tensors from s0, s1, s2, s3, s4
        features = self.backbone(x)
        s0, s1, s2, s3, s4 = features

        # 2. --- INPUT for Side-ViT-CNN (from CONV STEM) ---
        # Use the output of the first stage (s0) as a "learned patch embedding"
        # This is much better than feeding raw pixels.
        feats_cnn = F.interpolate(s0, size=(128, 128), mode='bilinear', align_corners=False)
        vit_out3 = self.side_vit_cnn(feats_cnn, K_value, Q_value)

        # 3. --- INPUT for Side-ViT-1 (from FPN) ---
        # Feed s1 and s2 into the FPN to get fused multi-scale features
        fpn_input = {'feat0': s1, 'feat1': s2}
        fpn_output = self.fpn(fpn_input)
        # Use the finest-resolution feature map from the FPN output
        feats_fpn = fpn_output['feat0']
        feats1 = self.proj_sv1(feats_fpn)
        feats1 = F.interpolate(feats1, size=(128, 128), mode='bilinear', align_corners=False)
        vit_out1 = self.sidevit1(feats1, K_value, Q_value)

        # 4. --- INPUT for Side-ViT-2 (from FINAL STAGE) ---
        feats2 = self.proj_sv2(s4)
        feats2 = F.interpolate(feats2, size=(128, 128), mode='bilinear', align_corners=False)
        vit_out2 = self.sidevit2(feats2, K_value, Q_value)

        # 5. --- FUSE ViT OUTPUTS using ATTENTION ---
        # The fusion module will learn to weigh the importance of each ViT path
        fused_vector = self.fusion_module(vit_out1, vit_out2, vit_out3)
        
        # 6. --- FINAL CLASSIFICATION ---
        logits = self.classifier_head(fused_vector)
        return logits
