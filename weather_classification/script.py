
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt

from data import WeatherDataset
from train import collate_fn, transform
from feature import *

data = '../../data/MWI/train_data'
labels = '../../data/MWI/train_labels'
train_data = WeatherDataset(data, labels, transform=transform)
train_dataloader = DataLoader(
        train_data, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=1
    )


for X,y in train_dataloader:
    print(X.size(), X.dtype)

    dark = dark_channel(X[0]).numpy()
    print(dark.shape)
    plt.imshow(dark, cmap='gray')
    plt.savefig('dark.jpg')
    plt.imshow(torch.movedim(X[0], 0, 2).numpy())
    plt.savefig('img3.jpg')
    break
