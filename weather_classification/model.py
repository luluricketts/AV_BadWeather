from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

def get_resnet50(params=None, n_classes=5):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, n_classes) # 4 for weatherdataset

    if params:
        model.load_state_dict(torch.load(params))

    return model