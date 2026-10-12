"""Model factory for the kidney tumor classifier (notebooks 03, 09 and 02)."""
import torch
import torch.nn as nn
from torchvision import models


class UNetClassifier(nn.Module):
    """Simplified 2D U-Net encoder used as a classifier (notebook 02)."""

    def __init__(self, num_classes=2, base=64):
        super().__init__()
        self.enc1 = self.conv_block(3, base)
        self.enc2 = self.conv_block(base, base * 2)
        self.enc3 = self.conv_block(base * 2, base * 4)
        self.enc4 = self.conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = self.conv_block(base * 8, base * 16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return self.fc(self.global_pool(b))


def _tv_weights(enum, pretrained):
    return enum.DEFAULT if pretrained else None


def _timm(name, num_classes, pretrained):
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def create_model(architecture='resnet18', num_classes=2, pretrained=True):
    """Build a classifier with a `num_classes`-way head."""
    if architecture == 'resnet18':
        model = models.resnet18(weights=_tv_weights(models.ResNet18_Weights, pretrained))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'resnet50':
        model = models.resnet50(weights=_tv_weights(models.ResNet50_Weights, pretrained))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=_tv_weights(models.EfficientNet_B0_Weights, pretrained))
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=_tv_weights(models.EfficientNet_B3_Weights, pretrained))
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == 'densenet121':
        model = models.densenet121(weights=_tv_weights(models.DenseNet121_Weights, pretrained))
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif architecture == 'vgg19':
        model = models.vgg19(weights=_tv_weights(models.VGG19_Weights, pretrained))
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    elif architecture == 'vit_b16':
        model = models.vit_b_16(weights=_tv_weights(models.ViT_B_16_Weights, pretrained))
        model.heads[0] = nn.Linear(model.heads[0].in_features, num_classes)
    elif architecture == 'xception':
        model = _timm('xception', num_classes, pretrained)
    elif architecture == 'mobilenetv3':
        model = _timm('mobilenetv3_large_100', num_classes, pretrained)
    elif architecture == 'vit':
        model = _timm('vit_base_patch16_224', num_classes, pretrained)
    elif architecture == 'unet':
        model = UNetClassifier(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown architecture: {architecture}. Choose from {AVAILABLE_MODELS}')
    return model


AVAILABLE_MODELS = ('resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b3', 'densenet121',
                    'vgg19', 'vit_b16', 'xception', 'mobilenetv3', 'vit', 'unet')


def load_model(architecture, checkpoint_path, num_classes=2, device='cpu'):
    """Create an architecture without pretrained weights and load a checkpoint into it."""
    model = create_model(architecture, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()
