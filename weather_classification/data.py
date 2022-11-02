import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
# TODO install torchvision on gpu server
# from torchvision import transforms

# transform = transforms.Compose([
#     transforms.RandomCrop(224,224),
#     transforms.RandomAffine(),
#     transforms.RandomRotation(),
#     transforms.ToTensor(),
# ])

class WeatherDataset(Dataset):

    def __init__(self, dir, transform=None):
        self.data_dir = os.path.join(dir, 'data')
        self.labels_dir = os.path.join(dir, 'labels')

        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):

        # what shape?
        im_str = idx + '.jpg'
        t_str = idx + '.txt'
        data_path = os.path.join(self.data_dir, im_str)
        label_path = os.path.join(self.labels_dir, t_str)

        img = np.asarray(Image.open(data_path))
        if self.transform:
            img = self.transform(img)
        with open(label_path, 'rb') as file:
            label = int(file.read())

        return img, label


img_dir = '../MWI-reformat'
dataset = WeatherDataset(img_dir, transform=None)
dataloader = DataLoader(dataset)