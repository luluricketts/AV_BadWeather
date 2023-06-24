import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import random
from PIL import Image as im
from collections import Counter
from experiment import AtmLight, DarkChannel
from scipy.ndimage import gaussian_filter

def hazy(airlight):
# name = '000004_10'
    f = "fe189115-352995ee"
    path = f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/bdd/{f}.jpg"
    img = plt.imread(path)
    
    depth_path = f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/bdd/{f}_depth.npy"
    depth_np = np.load(depth_path)
    depth_img = depth_np[0][0]
    depth_img = np.where(depth_img < depth_img.max()-depth_img.min(), depth_img-40, depth_img)
    depth_img = gaussian_filter(depth_img, sigma = 40)
    data = im.fromarray(depth_img)
    if data.mode != 'L':
        data = data.convert('L')
    data.save(f'/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/bdd/{f}_depth.jpg')
    depth_img_path = f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/bdd/{f}_depth.jpg"
    depth_img = plt.imread(depth_img_path)
    depth_img_3c = np.zeros_like(img)
    depth_img_3c[:,:,0] = depth_img
    depth_img_3c[:,:,1] = depth_img
    depth_img_3c[:,:,2] = depth_img
    norm_depth_img = depth_img_3c/255
    beta = 3.5
    trans = np.exp(-norm_depth_img*beta)
    A = airlight
    hazy = img*trans + A*(1-trans)
    hazy = np.array(hazy, dtype=np.uint8)
    cv2.imwrite(f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/results/bdd/{f}.jpg", cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB))

