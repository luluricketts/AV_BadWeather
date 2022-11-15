import os
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class WeatherDataset(Dataset):

    def __init__(self, data_path, target_path, transform=None):
        self.data_dir = data_path
        self.labels_dir = target_path
        
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):

        im_str = f'{idx}.jpg'
        t_str = f'{idx}.txt'
        data_path = os.path.join(self.data_dir, im_str)
        label_path = os.path.join(self.labels_dir, t_str)

        img = Image.open(data_path)
    
        if self.transform:
            img = self.transform(img)
        with open(label_path, 'rb') as file:
            label = int(file.read())
            label = torch.tensor(label)
        
        return img, label


class WeatherDataset2(Dataset):

    def __init__(self, data_dir, data_json, transform=None):
        self.data_dir = data_dir
        with open(data_json) as file:
            self.metadata = json.load(file)
        
        self.transform = transform

    def __len__(self):
        return len(self.metadata) 

    def __getitem__(self, idx):

        filename = self.metadata[str(idx)]['path'].split('/')[1]
        img_path = os.path.join(self.data_dir, filename)

        img = Image.open(img_path)
        label = int(self.metadata[str(idx)]['label'])
        label = torch.tensor(label)

        source = self.metadata[str(idx)]['source']
        
        if self.transform:
            img = self.transform(img)

        return img, label, source

