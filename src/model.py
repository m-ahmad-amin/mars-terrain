from torch import nn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

from .config import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights, aux_loss=True)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model
