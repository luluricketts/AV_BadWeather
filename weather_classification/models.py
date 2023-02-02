from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature import get_all_features


# TODO add ViT model


def get_resnet50(params=None, n_classes=4):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, n_classes) 

    if params:
        model.load_state_dict(torch.load(params))

    return model



class ResNet(nn.Module):
    def __init__(self, params=None, n_classes=4, feature_len=45):
        super(ResNet, self).__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(1024 + feature_len, 64) 
        self.fc2 = nn.Linear(64, n_classes)


        if params:
            try:
                self.resnet.load_state_dict(torch.load(params))
            except:
                # load and freeze params
                self.load_nonmatched_weights(torch.load(params))
                for name,param in self.resnet.named_parameters():
                    if 'fc' in name: continue
                    param.requires_grad = False
    

    def forward(self, x):
        res_features = self.resnet(x)
        features = get_all_features(x)
        x = torch.cat((res_features, features), dim=1)
        x = F.relu(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)
        return output
    
    def load_nonmatched_weights(self, weights):

        model_weights = self.resnet.state_dict()
        for name, param in weights.items():
            if name not in model_weights or 'fc' in name:
                continue
            model_weights[name].copy_(param.data)