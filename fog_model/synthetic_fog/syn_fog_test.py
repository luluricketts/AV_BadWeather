import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import random
from PIL import Image as im

path = f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/monodepth_sample/val01-25-20-1-FRONT.jpg"
img = plt.imread(path)
depth_np = np.load("/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/kitti_sample/000000_1_depth.npy")
depth_img_path = f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/monodepth_sample/val01-25-20-1-FRONT_depth.jpg"
depth_img = plt.imread(depth_img_path)
# depth_img = depth_np[0][0]
# data = im.fromarray(depth_img)
# if data.mode != 'L':
#     data = data.convert('L')
# data.save('/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/kitti_sample/000000_1_depth.jpg')
# depth_img_path = f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/kitti_sample/000000_1_depth.jpg"
# depth_img = plt.imread(depth_img_path)

depth_img_3c = np.zeros_like(img)
depth_img_3c[:,:,0] = depth_img
depth_img_3c[:,:,1] = depth_img
depth_img_3c[:,:,2] = depth_img
norm_depth_img = depth_img_3c/255
beta = 0.5

# For blob creation

(centerX, centerY) = (list(zip(*np.where(depth_img == 255)))[0])
beta = np.ones_like(norm_depth_img)*1.4

# centerX = 640
# centerY = 960
radius = 500
for i in range(len(beta)):
    for j in range(len(beta[i])):
        dist = ((i - centerX)**2 + (j - centerY)**2)
        if dist <= radius**2:
            beta[i,j,:]=(1.2+(((radius+1)/(dist+200))*100))

trans = np.exp(-norm_depth_img*beta)
A = 255
hazy = img*trans + A*(1-trans)
hazy = np.array(hazy, dtype=np.uint8)
cv2.imwrite(f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/results/result_monodepth_fog_blob.jpg", cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB))