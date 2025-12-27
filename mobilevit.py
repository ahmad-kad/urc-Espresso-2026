"""
Simplified MobileNetV3 Object Detector
Streamlined version for fast training - removed ViT components
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3Backbone(nn.Module):
    """Simplified MobileNetV3 backbone"""

    def __init__(self, pretrained=True, input_size=224):
        super(MobileNetV3Backbone, self).__init__()

        # Load MobileNetV3-Small
        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None

        self.backbone = mobilenet_v3_small(weights=weights)
        self.features = self.backbone.features

    def forward(self, x):
        """Extract final feature map only"""
        x = self.features(x)  # Run through all layers
        return x


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


class MobileVitDetector(nn.Module):
    """Simplified MobileNetV3 object detector (ViT components removed for speed)"""

    def __init__(self, num_classes, pretrained=True, input_size=224):
        super(MobileVitDetector, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.backbone = MobileNetV3Backbone(pretrained=pretrained, input_size=input_size)

        # Initialize with dummy input to determine feature channels
        dummy_input = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            dummy_features = self.backbone(dummy_input)

        # Get feature channels from the final feature map
        self.feature_channels = dummy_features.shape[1]

        # Single detection head on the final feature map
        self.detector = DetectionHead(self.feature_channels, num_classes, num_anchors=9)

    def forward(self, x):
        # Extract features from backbone
        final_feature = self.backbone(x)  # Shape: [batch, channels, h, w]

        # Apply detection head
        cls_logits, bbox_preds = self.detector(final_feature)

        return cls_logits, bbox_preds


def create_mobilevit_detector(num_classes, pretrained=True, input_size=224):
    """Create simplified MobileNetV3 object detector"""
    model = MobileVitDetector(
        num_classes=num_classes,
        pretrained=pretrained,
        input_size=input_size
    )
    return model


# Test the simplified model
if __name__ == "__main__":
    # Test model creation
    model = create_mobilevit_detector(num_classes=6, input_size=160)
    print("Simplified MobileNetV3 detector created successfully")

    # Test forward pass
    x = torch.randn(1, 3, 160, 160)
    cls_logits, bbox_preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Classification output shape: {cls_logits.shape}")
    print(f"Regression output shape: {bbox_preds.shape}")
    print("Forward pass successful!")