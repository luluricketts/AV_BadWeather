from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature import get_all_features, hog


class ViT(nn.Module):
    def __init__(self, eng_feats:bool, freeze_backbone:bool, params=None, n_classes=4, n_features=3):
        super(ViT, self).__init__()

        self.eng_feats = eng_feats # use engineered features
        self.vit = models.vit_b_16(weights=models.vision_transformer.ViT_B_16_Weights.DEFAULT)

        if self.eng_feats:
            self.vit.conv_proj = nn.Conv2d(3+n_features, 768, kernel_size=(16,16), stride=(16,16))
        
        self.vit.heads = nn.Sequential(
            nn.Linear(768, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

        if freeze_backbone:
            for n,param in self.vit.named_parameters():
                if 'heads' in n:
                    continue
                param.requires_grad = False

        if params:
            self.vit.load_state_dict(torch.load(params))

    
    def forward(self, x):
        
        if self.eng_feats:
            features = get_all_features(x, hist=False)
            x = torch.concat((x, features), dim=1)
        output = torch.sigmoid(self.vit(x))
        return output



class FeaturesOnly(nn.Module):
    def __init__(self, eng_feats=True, freeze_backbone=False, params=None, n_classes=4):
        super(FeaturesOnly, self).__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.classification_head = nn.Sequential(
            nn.Linear(1200, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
        self.classification_fog_head = nn.Sequential(
            nn.Linear(1200, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        all_feats, hist_feats = get_all_features(x, hist=False)
        feats = self.backbone(all_feats)
        feats = torch.cat((feats, hist_feats), dim=1)
        out1 = self.classification_head(feats)
        out2 = self.classification_fog_head(feats)
        return out1, out2

class ResNet(nn.Module):
    def __init__(self, eng_feats:bool, freeze_backbone:bool, params=None, n_classes=4, feature_len=45):
        super(ResNet, self).__init__()

        self.eng_feats = eng_feats
        if not self.eng_feats: feature_len = 0

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.classification_head = nn.Sequential(
            nn.Linear(1000 + feature_len, 512),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(128, 32),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(32, n_classes)
        )
        self.fog_classification_head = nn.Sequential(
                nn.Linear(1000 + feature_len, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 4)
        )
        # can change number of resnet features frozen
        if freeze_backbone:
            # for param in self.resnet.layer1.parameters():
            #     param.requires_grad = False
            # for param in self.resnet.layer2.parameters():
            #     param.requires_grad = False
            # for param in self.resnet.layer3.parameters():
            #     param.requires_grad = False
            for param in self.resnet.parameters():
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

        all_features = self.resnet(x)
        if self.eng_feats:
            features = get_all_features(x)
            all_features = torch.cat((all_features, features), dim=1)
        x = self.classification_head(all_features)
        y = self.fog_classification_head(all_features)
        output1 = torch.sigmoid(x)
        output2 = torch.sigmoid(y)
        return output1, output2
    
    def load_nonmatched_weights(self, weights):

        model_weights = self.resnet.state_dict()
        for name, param in weights.items():
            if name not in model_weights or 'fc' in name:
                continue
            model_weights[name].copy_(param.data)

