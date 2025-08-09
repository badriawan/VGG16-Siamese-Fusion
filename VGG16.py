import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG16FeatureExtractor(nn.Module):
    """VGG16-based feature extractor with improved error handling"""
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        try:
            print("Loading VGG16 pretrained model...")
            vgg16 = models.vgg16(pretrained=True)
            print("✓ VGG16 loaded successfully")

            self.features = vgg16.features

            # Create adaptive classifier to handle different input sizes
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 1024),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(1024, 512)
            )
            print("✓ Feature extractor initialized")

        except Exception as e:
            print(f"✗ Error initializing VGG16: {e}")
            raise

    def forward(self, x):
        try:
            x = self.features(x)
            x = self.classifier(x)
            return x
        except Exception as e:
            print(f"Error in VGG16 forward pass: {e}")
            print(f"Input shape: {x.shape}")
            raise
