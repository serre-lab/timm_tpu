import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

import timm
from timm.models._registry import register_model

__all__ = []

@register_model
def new_resnet50(pretrained = False, **kwargs):
    if pretrained:
        model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = resnet50()
    return model


