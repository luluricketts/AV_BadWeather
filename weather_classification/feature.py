import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as tvF

# ALL FUNCS TAKE IN 3x224x224 tensor

def batch_histogram(features, bins, gt0=False):
    if gt0:
        hist = [list(torch.histogram(batch[batch>0], bins=bins, density=True)[0]) for batch in features]
    else:
        hist = [list(torch.histogram(batch, bins=bins, density=True)[0]) for batch in features]
    hist = torch.from_numpy(np.array(hist))
    
    return hist

def dark_channel(img, hist):
    # img tensor of shape 3x224x224
    # returns hist
    dark = nn.MaxPool2d((3,3), stride=1, padding=1)
    dark_chan = -dark(-img)
    dark_c = torch.min(dark_chan, dim=1).values

    if hist:
        dark_hist = batch_histogram(dark_c, bins=25)
        return dark_hist
    return dark_c


def hog(img):
    img = torch.movedim(img, 1, 3).numpy()
    img = [255 * (i - np.min(i)) / (np.max(i) - np.min(i)) for i in img]
    img = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in img]
    low_freq = [cv2.GaussianBlur(i, (3, 3), 9) for i in img]
    high_freq = [(img[i] - low_freq[i]).astype(np.uint8) for i in range(len(img))] 

    # thres = [cv2.threshold(hi, 120, 255, cv2.THRESH_BINARY)[1].astype(np.uint8) for hi in high_freq]
    hog = cv2.HOGDescriptor()
    hist = [hog.compute(t) for t in high_freq]
    hist = torch.tensor(np.asarray(hist))
    hist = batch_histogram(hist, bins=200, gt0=True)

    return hist


def saturation(img, hist):
    """
    returns 10-dim feature vector for saturation
    """
    img = torch.movedim(img, 1, 3).numpy()
    img_hsv = [cv2.cvtColor(i, cv2.COLOR_RGB2HSV) for i in img]
    img_hsv = torch.from_numpy(np.array(img_hsv))

    if hist:
        sat_hist = batch_histogram(img_hsv[:,:,:,1], bins=10)
        return sat_hist
    return img_hsv[:,:,:,1]


def local_contrast(img, ksize, hist):
    
    windows = F.unfold(img, kernel_size=ksize, stride=1)
    var = torch.var(windows, unbiased=False, dim=1)
    out = F.fold(var, (img.shape[2]-ksize+1, img.shape[3]-ksize+1), kernel_size=1)
    
    if hist:
        contr_hist = batch_histogram(out, bins=10)
        return contr_hist
    return F.pad(out, (5, 5, 5, 5), mode='constant', value=0)


def get_all_features(img, hist=True):
    img = img.cpu()

    dark = dark_channel(img, hist)
    sat = saturation(img, hist)
    contrast = local_contrast(img, 11, hist)
    hog_feat = hog(img)

    if hist:
        features = torch.concat((dark, sat, contrast, hog), dim=1)
        return features.to('cuda')
    else:
        dark = dark.unsqueeze(1)
        sat = sat.unsqueeze(1)
        contrast = contrast.unsqueeze(1)
        features = torch.concat((dark, sat, contrast), dim=1)
        hist_features = hog_feat
        # print(hist_features, hist_features.shape)
        return features.to('cuda'), hist_features.to('cuda')


# import skimage
# import os
# import matplotlib.pyplot as plt 

# base = 'test2'
# rain = os.path.join(base, 'rain.jpg')
# clear = os.path.join(base, 'clear.jpg')
# snow = os.path.join(base, 'snow.jpg')
# fog = os.path.join(base, 'fog.jpg')

# rimg = torch.tensor(skimage.io.imread(rain))
# rimg = torch.movedim(rimg, 2, 0).float().unsqueeze(0)
# dark, sat, con, hist = get_all_features(rimg, hist=False)
# dark = dark.squeeze()
# sat = sat.squeeze()
# con = con.squeeze()
# skimage.io.imsave(os.path.join(base, 'rain-dark.jpg'), dark)
# skimage.io.imsave(os.path.join(base, 'rain-sat.jpg'), sat)
# skimage.io.imsave(os.path.join(base, 'rain-con.jpg'), con)
# skimage.io.imsave(os.path.join(base, 'rain-hog.jpg'), hist[0])
# # plt.hist(hist[hist>0], bins=200, alpha=0.5)
# # plt.savefig(os.path.join(base, 'rain-hist.jpg'))

# cimg = torch.tensor(skimage.io.imread(clear))
# cimg = torch.movedim(cimg, 2, 0).float().unsqueeze(0)
# dark, sat, con, hist = get_all_features(cimg, hist=False)
# dark = dark.squeeze()
# sat = sat.squeeze()
# con = con.squeeze()
# skimage.io.imsave(os.path.join(base, 'clear-dark.jpg'), dark)
# skimage.io.imsave(os.path.join(base, 'clear-sat.jpg'), sat)
# skimage.io.imsave(os.path.join(base, 'clear-con.jpg'), con)
# skimage.io.imsave(os.path.join(base, 'clear-hog.jpg'), hist[0])
# # plt.hist(hist[hist>0], bins=200, alpha=0.5)
# # plt.savefig(os.path.join(base, 'clear-hist.jpg'))

# simg = torch.tensor(skimage.io.imread(snow))
# simg = torch.movedim(simg, 2, 0).float().unsqueeze(0)
# dark, sat, con, hist = get_all_features(simg, hist=False)
# dark = dark.squeeze()
# sat = sat.squeeze()
# con = con.squeeze()
# skimage.io.imsave(os.path.join(base, 'snow-dark.jpg'), dark)
# skimage.io.imsave(os.path.join(base, 'snow-sat.jpg'), sat)
# skimage.io.imsave(os.path.join(base, 'snow-con.jpg'), con)
# skimage.io.imsave(os.path.join(base, 'snow-hog.jpg'), hist[0])
# # plt.hist(hist[hist>0], bins=200, alpha=0.5)
# # plt.savefig(os.path.join(base, 'snow-hist.jpg'))

# fimg = torch.tensor(skimage.io.imread(fog))
# fimg = torch.movedim(fimg, 2, 0).float().unsqueeze(0)
# dark, sat, con, hist = get_all_features(fimg, hist=False)
# dark = dark.squeeze()
# sat = sat.squeeze()
# con = con.squeeze()
# skimage.io.imsave(os.path.join(base, 'fog-dark.jpg'), dark)
# skimage.io.imsave(os.path.join(base, 'fog-sat.jpg'), sat)
# skimage.io.imsave(os.path.join(base, 'fog-con.jpg'), con)
# skimage.io.imsave(os.path.join(base, 'fog-hog.jpg'), hist[0])
# # plt.hist(hist[hist>0], bins=200, alpha=0.5)
# # plt.savefig(os.path.join(base, 'fog-hist.jpg'))

