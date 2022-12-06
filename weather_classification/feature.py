import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ALL FUNCS TAKE IN 3x224x224 tensor

def batch_histogram(features, bins):
    hist = [list(torch.histogram(batch, bins=bins, density=True)[0]) for batch in features]
    hist = torch.from_numpy(np.array(hist))
    
    return hist

def dark_channel(img):
    # img tensor of shape 3x224x224
    # returns hist
    dark = nn.MaxPool2d((3,3), stride=1, padding=1)
    dark_chan = -dark(-img)
    dark_c = torch.min(dark_chan, dim=1).values

    dark_hist = batch_histogram(dark_c, bins=25)
    return dark_hist


def hog(img):
    ...


def saturation(img):
    """
    returns 10-dim feature vector for saturation
    """
    img = torch.movedim(img, 1, 3).numpy()
    img_hsv = [cv2.cvtColor(i, cv2.COLOR_RGB2HSV) for i in img]
    img_hsv = torch.from_numpy(np.array(img_hsv))

    sat_hist = batch_histogram(img_hsv[:,:,:,1], bins=10)
    return sat_hist


def local_contrast(img, ksize=11):
    
    windows = F.unfold(img, kernel_size=ksize, stride=1)
    var = torch.var(windows, unbiased=False, dim=1)
    out = F.fold(var, (img.shape[2]-ksize+1, img.shape[3]-ksize+1), kernel_size=1)
    
    contr_hist = batch_histogram(out, bins=10)
    return contr_hist


def get_all_features(img):
    img = img.cpu()

    dark = dark_channel(img)
    sat = saturation(img)
    contrast = local_contrast(img)
    
    features = torch.concat((dark, sat, contrast), dim=1)
    return features.to('cuda')




# def test_features():

#     from data import WeatherDataset
#     from train import transform, collate_fn
#     from torch.utils.data import DataLoader
#     import skimage
#     import os

#     save_path = './test_imgs'

#     data_dir = '../../data/weather_classification/data'
#     train = '../../data/weather_classification/train.json'
#     dataset = WeatherDataset(data_dir, train, transform=transform)
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=1, 
#         shuffle=True, 
#         collate_fn=collate_fn,
#         num_workers=1
#     )


#     for X,y,_ in dataloader:
#         X = torch.squeeze(X, dim=0)

#         f = local_contrast(X)
#         print(f, f.shape, f.sum(), type(f))
#         break


# test_features()
