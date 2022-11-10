import torch.nn as nn
import torch

def dark_channel(img):
    # img of shape 3x224x224
    dark = nn.MaxPool2d((15,15), stride=1, padding=1)
    dark_c = torch.min(-dark(-img), dim=0).values
    return dark_c