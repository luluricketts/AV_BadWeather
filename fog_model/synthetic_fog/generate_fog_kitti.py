import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

folder_path = "/home/roshini/AV_BadWeather/TransWeather/data/kitti_depth/stereo/training/image_2/"
file_txt = open("/home/roshini/AV_BadWeather/TransWeather/data/kitti_depth/stereo/training/foggy_image_2/data.txt","w")
file_txt.write("Filename     Airlight                  Beta\n")
n = 0
for file in os.listdir(folder_path):
    f, ext = os.path.splitext(file)
    if ext=='.png':
        path = f"/home/roshini/AV_BadWeather/TransWeather/data/kitti_depth/stereo/training/image_2/{f}.png"
        img = plt.imread(path)*255
        A = min(np.abs(np.random.normal(210, 15, 1))[0], 255)
        depth_img_path = f"/home/roshini/AV_BadWeather/TransWeather/data/kitti_depth/stereo/training/depth_image_2/{f}.png"
        if os.path.exists(depth_img_path):
            depth_img = plt.imread(depth_img_path)
            if depth_img.shape[0]!=375 or depth_img.shape[1]!=1242 or img.shape[0]!=375 or img.shape[1]!=1242:
                continue
            depth_img_3c = np.zeros_like(img)
            depth_img_3c[:,:,0] = depth_img
            depth_img_3c[:,:,1] = depth_img
            depth_img_3c[:,:,2] = depth_img
            norm_depth_img = depth_img_3c
            beta = np.abs(np.random.normal(2.5, 1, 1))[0]
            trans = np.exp(-norm_depth_img*beta)
            hazy = img*trans + A*(1-trans)
            hazy = np.array(hazy, dtype=np.uint8)
            cv2.imwrite(f"/home/roshini/AV_BadWeather/TransWeather/data/kitti_depth/stereo/training/foggy_image_2/{f}.png", cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB))
            file_txt.write(f"{f}    {A}         {beta}\n")
            n = n + 1
file_txt.close()
print(n)
