import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

#fname = "2011_09_26_drive_0036_sync_image_0000000014_image_03"
#path = f"data/image_data/{fname}.png"
fname = "2011_09_26_drive_0002_sync_image_0000000005_image_02"
path = f"data/image_data/{fname}.png"

img_plot = plt.imread(path)
img = img_plot*255
#img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

#fname_depth = "2011_09_26_drive_0036_sync_groundtruth_depth_0000000014_image_03"
#depth_path = f"data/depth_data/{fname_depth}.png"
fname_depth = "2011_09_26_drive_0002_sync_image_0000000005_image_02_disp"
depth_path = f"data/image_data/{fname_depth}.jpeg"
depth_img = plt.imread(depth_path)
depth_img = color.rgb2gray(depth_img)
#depth_img = 255 - cv2.normalize(depth_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
depth_img = 255 - (depth_img*255)
depth_img_3c = np.zeros_like(img)
depth_img_3c[:,:,0] = depth_img
depth_img_3c[:,:,1] = depth_img
depth_img_3c[:,:,2] = depth_img

beta=0.5
norm_depth_img = depth_img_3c/255
trans = np.exp(-norm_depth_img*beta)

beta2=2.2
trans2 = np.exp(-norm_depth_img*beta2)

A = 255
hazy = img*trans + A*(1-trans)
hazy = np.array(hazy, dtype=np.uint8)

hazy2 = img*trans2 + A*(1-trans2)
hazy2 = np.array(hazy2, dtype=np.uint8)
plt.figure(figsize=(16,18))
plt.subplot(411), plt.imshow(img_plot)
plt.subplot(412), plt.imshow(depth_img, cmap="gray")
plt.subplot(413), plt.imshow(hazy)
plt.subplot(414), plt.imshow(hazy2)

plt.savefig('result.png')
plt.show()