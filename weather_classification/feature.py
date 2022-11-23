import cv2
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

# ALL FUNCS TAKE IN 3x224x224 tensor
# TODO add to dataloader

def dark_channel(img):
    # img tensor of shape 3x224x224
    dark = nn.MaxPool2d((3,3), stride=1, padding=1)
    dark_chan = -dark(-img)
    dark_c = torch.min(dark_chan, dim=0).values
    return dark_c


def hog(img):
    ...


def saturation(img, bins=10):
    """
    returns 10-dim feature vector for saturation
    TODO (?) return as tensor?
    """
    img = torch.movedim(img, 0, 2).numpy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    norm_sat = (img_hsv[:,:,1] - np.min(img_hsv[:,:,1])) /\
            (np.max(img_hsv[:,:,1]) - np.min(img_hsv[:,:,1]))

    sat_hist = np.histogram(norm_sat, bins=bins, density=True)[0]
    return sat_hist


def local_contrast(img, ksize=11, bins=10):
    
    windows = f.unfold(img, kernel_size=ksize, stride=1)

    var = torch.var(windows, unbiased=False, dim=0)
    var = torch.unsqueeze(var, dim=0)

    out = f.fold(var, (img.shape[1]-ksize+1, img.shape[2]-ksize+1), kernel_size=1)
    out = f.pad(out, (2,2,2,2))
    
    hist = np.histogram(torch.squeeze(out, dim=0), bins=bins, density=True)[0]
    return hist # check if this is return probability dist




def test_features():

    from data import WeatherDataset
    from train import transform, collate_fn
    from torch.utils.data import DataLoader
    import skimage
    import os

    save_path = './test_imgs'

    data_dir = '../../data/weather_classification/data'
    train = '../../data/weather_classification/train.json'
    dataset = WeatherDataset(data_dir, train, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=1
    )


    for X,y,_ in dataloader:
        X = torch.squeeze(X, dim=0)

        l = local_contrast(X)
        print(l)

        d = dark_channel(X)
        s = saturation(X)

        print('s', s)
        # # skimage.io.imsave(os.path.join(save_path, 'sat.png'), s)

        # print('d', d.shape)
        # skimage.io.imsave(os.path.join(save_path, 'dark.png'), d)
        
        print('x', X.shape)
        skimage.io.imsave(os.path.join(save_path, 'orig.png'), torch.movedim(X, 0, 2))
        break


test_features()
