from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature import get_all_features


class ViT(nn.Module):
    def __init__(self, params=None, n_classes=4, n_features=3):
        super(ViT, self).__init__()

        self.vit = models.vit_b_16(weights=models.vision_transformer.ViT_B_16_Weights.DEFAULT)
        self.vit.conv_proj = nn.Conv2d(3, 768, kernel_size=(16,16), stride=(16,16))
        self.vit.heads = nn.Sequential(
            nn.Linear(768, 600),
            nn.Tanh(),
            nn.Linear(600, 300),
            nn.Tanh(),
            nn.Linear(300, 150),
            nn.Tanh(),
            nn.Linear(150, 32),
            nn.Tanh(),
            nn.Linear(32, n_classes)
        )

        for n,param in self.vit.named_parameters():
            if 'heads' in n:
                continue
            param.requires_grad = False


    
    def forward(self, x):
  
        # features = get_all_features(x, hist=False)
        # x = torch.concat((x, features), dim=1)
        output = torch.sigmoid(self.vit(x))
        return output






class Resnet_nofeatures(nn.Module):

    def __init__(self, params=None, n_classes=4):
        super(Resnet_nofeatures, self).__init__()    
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.classification_head = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

        for param in self.model.parameters():
            param.requires_grad = False

        if params:
            model.load_state_dict(torch.load(params))

    def forward(self, x):
        x = self.model(x)
        x = self.classification_head(x)
        out = torch.sigmoid(x)

        return out



class ResNet(nn.Module):
    def __init__(self, params=None, n_classes=4, feature_len=45):
        super(ResNet, self).__init__()

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(1024 + feature_len, 512) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, n_classes)

        # use ImageNet features for layers 1/2/3
        for param in self.resnet.layer1.parameters():
            param.requires_grad = False
        for param in self.resnet.layer2.parameters():
            param.requires_grad = False
        for param in self.resnet.layer3.parameters():
            param.requires_grad = False

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
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        output = torch.sigmoid(self.fc4(x))
        return output
    
    def load_nonmatched_weights(self, weights):

        model_weights = self.resnet.state_dict()
        for name, param in weights.items():
            if name not in model_weights or 'fc' in name:
                continue
            model_weights[name].copy_(param.data)