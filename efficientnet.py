"""
EfficientNet-based Object Detector
Custom implementation for object detection using EfficientNet backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import math


class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone with feature extraction"""

    def __init__(self, pretrained=True):
        super(EfficientNetBackbone, self).__init__()

        # Load EfficientNet-B0
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None

        self.backbone = efficientnet_b0(weights=weights)

        # Remove the classifier
        self.features = self.backbone.features

        # Feature map sizes at different levels
        self.feature_channels = [16, 24, 40, 112, 1280]  # EfficientNet-B0 feature channels

    def forward(self, x):
        """Extract multi-scale features"""
        features = []

        # EfficientNet feature extraction
        x = self.features[0](x)  # First conv + blocks
        features.append(x)  # Level 1: 16 channels

        x = self.features[1](x)  # Block 2
        features.append(x)  # Level 2: 24 channels

        x = self.features[2](x)  # Block 3
        features.append(x)  # Level 3: 40 channels

        x = self.features[3](x)  # Block 4
        features.append(x)  # Level 4: 112 channels

        x = self.features[4](x)  # Block 5
        x = self.features[5](x)  # Block 6
        x = self.features[6](x)  # Block 7
        x = self.features[7](x)  # Final conv
        x = self.features[8](x)  # Adaptive avg pool
        features.append(x)  # Level 5: 1280 channels

        return features


class DetectionHead(nn.Module):
    """Detection head for bounding box regression and classification"""

    def __init__(self, in_channels, num_classes, num_anchors=9):
        super(DetectionHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * num_classes, 1)
        )

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * 4, 1)  # 4 coordinates: x,y,w,h
        )

    def forward(self, x):
        cls_logits = self.cls_head(x)
        bbox_preds = self.reg_head(x)

        # Reshape for anchor-based detection
        batch_size, _, height, width = cls_logits.shape

        # Classification: [batch, num_classes, height, width, num_anchors]
        cls_logits = cls_logits.view(batch_size, self.num_classes, -1, height, width)
        cls_logits = cls_logits.permute(0, 1, 3, 4, 2).contiguous()
        cls_logits = cls_logits.view(batch_size, -1, self.num_classes)

        # Regression: [batch, 4, height, width, num_anchors]
        bbox_preds = bbox_preds.view(batch_size, 4, -1, height, width)
        bbox_preds = bbox_preds.permute(0, 1, 3, 4, 2).contiguous()
        bbox_preds = bbox_preds.view(batch_size, -1, 4)

        return cls_logits, bbox_preds


class EfficientNetDetector(nn.Module):
    """Complete EfficientNet-based object detector"""

    def __init__(self, num_classes, pretrained=True, attention_layers=None):
        super(EfficientNetDetector, self).__init__()
        self.num_classes = num_classes
        self.backbone = EfficientNetBackbone(pretrained=pretrained)

        # Feature Pyramid Network for multi-scale features
        self.fpn = FeaturePyramidNetwork(self.backbone.feature_channels)

        # Detection heads for different scales
        self.heads = nn.ModuleList([
            DetectionHead(256, num_classes) for _ in range(5)  # 5 feature levels
        ])

        # Add attention layers if specified
        self.attention_layers = attention_layers or []
        if self.attention_layers:
            self._add_attention_layers()

    def _add_attention_layers(self):
        """Add CBAM attention to specified layers"""
        try:
            from .attention_modules import CBAM

            for layer_name in self.attention_layers:
                if hasattr(self.backbone.features, layer_name):
                    layer = getattr(self.backbone.features, layer_name)
                    # Wrap the layer with CBAM
                    setattr(self.backbone.features, layer_name,
                           nn.Sequential(layer, CBAM(layer.out_channels)))
        except ImportError:
            print("Warning: CBAM attention modules not available")

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Build feature pyramid
        fpn_features = self.fpn(features)

        # Apply detection heads
        cls_logits_list = []
        bbox_preds_list = []

        for i, feature in enumerate(fpn_features):
            cls_logits, bbox_preds = self.heads[i](feature)
            cls_logits_list.append(cls_logits)
            bbox_preds_list.append(bbox_preds)

        # Concatenate predictions from all scales
        cls_logits = torch.cat(cls_logits_list, dim=1)
        bbox_preds = torch.cat(bbox_preds_list, dim=1)

        return cls_logits, bbox_preds


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""

    def __init__(self, in_channels_list):
        super(FeaturePyramidNetwork, self).__init__()
        self.in_channels_list = in_channels_list

        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, 256, 1) for in_channels in in_channels_list
        ])

        # Top-down pathway
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        # Build laterally connected features
        lateral_features = []
        for i, feature in enumerate(features):
            lateral_features.append(self.lateral_convs[i](feature))

        # Build top-down pathway
        fpn_features = [lateral_features[-1]]  # Start with highest level

        for i in range(len(lateral_features) - 2, -1, -1):
            # Upsample
            upsampled = F.interpolate(fpn_features[-1], size=lateral_features[i].shape[-2:], mode='nearest')

            # Add lateral connection
            fpn_feature = lateral_features[i] + upsampled

            # Apply conv
            fpn_feature = self.fpn_convs[i](fpn_feature)

            fpn_features.append(fpn_feature)

        # Reverse to match input order (P2, P3, P4, P5, P6)
        fpn_features.reverse()

        return fpn_features


def create_efficientnet_detector(num_classes, pretrained=True, attention_layers=None):
    """Create EfficientNet-based object detector"""
    model = EfficientNetDetector(
        num_classes=num_classes,
        pretrained=pretrained,
        attention_layers=attention_layers
    )
    return model


# Test the model
if __name__ == "__main__":
    # Test model creation
    model = create_efficientnet_detector(num_classes=6)
    print("EfficientNet detector created successfully")

    # Test forward pass
    x = torch.randn(1, 3, 416, 416)
    cls_logits, bbox_preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Classification output shape: {cls_logits.shape}")
    print(f"Regression output shape: {bbox_preds.shape}")








