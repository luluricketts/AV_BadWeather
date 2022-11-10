import os

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
        plt.imshow(np.array(img))
        plt.savefig('img.jpg')

        if self.transform:
            img = self.transform(img)
        with open(label_path, 'rb') as file:
            label = int(file.read())
            label = torch.tensor(label)

        plt.imshow(torch.movedim(img,0,2).numpy())
        plt.savefig('img2.jpg')
        
        return img, label

