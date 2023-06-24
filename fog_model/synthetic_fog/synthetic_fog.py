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

name = '000004_10'
# path = f"/home/roshini/AV_BadWeather/TransWeather/data/kitti_depth/stereo/training/image_2/{name}.png"
#thresh_path = f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/results/thresh/{name}.png"
pathai = "/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/bdd/"
for file in os.listdir(pathai):
    f, ext = os.path.splitext(file)
    if ext=='.jpg':
        path = f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/bdd/{f}.jpg"
        img = plt.imread(path)
        # src = cv2.imread(path)
        # I = src.astype('float64')/255
        # dark = DarkChannel(I,3)
        # A = AtmLight(I,dark)+20
        A = np.abs(np.random.normal(217.5, 38.75, 1))
        # #thresh_img = plt.imread(thresh_path)
        # thresh_img_3d = np.zeros_like(img)
        # thresh_img_3d[:,:,0] = thresh_img 
        # thresh_img_3d[:,:,1] = thresh_img 
        # thresh_img_3d[:,:,2] = thresh_img 
        img = img
        # b, g, r = cv2.split(img)
        # for i in range(len(b)):
        #     for j in range(len(b[0])):
        #         if b[i][j]<g[i][j] and b[i][j]<r[i][j]:
        #             b[i][j] = b[i][j]+30
        #         elif g[i][j]<b[i][j] and g[i][j]<r[i][j]:
        #             g[i][j] = g[i][j]
        #         elif r[i][j]<g[i][j] and r[i][j]<b[i][j]:
        #             r[i][j] = r[i][j]+30
        # new_img = cv2.merge([b, g, r])
        # print(img, new_img)
        #img = new_img*thresh_img_3d + img*(1-thresh_img_3d)
        img = img#cv2.merge([b, g, r])
        # out = cv2.addWeighted( img, 5, img, 0, 0.2)
        # cv2.imwrite(f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/results/experiment/{name}_try.jpg", cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

                
                
        # recounted = Counter(dc)
        # print(recounted)
        depth_path = f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/bdd/{f}_depth.npy"
        if os.path.exists(depth_path):
            depth_np = np.load(depth_path)
        else:
            continue
        # depth_img_path = f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/bdd/fde2b675-044dabc5_depth.jpg"
        # depth_img = plt.imread(depth_img_path)
        depth_img = depth_np[0][0]

        # depth_img = np.where(depth_img > depth_img.max()-depth_img.min(), depth_img-5, depth_img)
        depth_img = np.where(depth_img < depth_img.max()-depth_img.min(), depth_img-40, depth_img)
        depth_img = gaussian_filter(depth_img, sigma = 40)
        # depth_img = ((depth_img - depth_img.min()) / (depth_img.max()-depth_img.min())) 
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
        # depth_img_3c[:,:,3] = depth_img
        norm_depth_img = depth_img_3c/255
        beta = np.abs(np.random.normal(3, 1, 1))
        '''
        # For blob creation

        (centerX, centerY) = (list(zip(*np.where(depth_img == 255)))[0])
        beta = np.ones_like(norm_depth_img)*1.7
        print(beta.shape)
        # centerX = 640
        # centerY = 960
        radius = 200
        for i in range(len(beta)):
            for j in range(len(beta[i])):
                dist = ((i - centerX)**2 + (j - centerY)**2)
                if dist <= radius**2:
                    beta[i,j,:]=(1.2+(((radius+1)/(dist+200))*100))
        '''
        trans = np.exp(-norm_depth_img*beta)
        hazy = img*trans + A*(1-trans)
        hazy = np.array(hazy, dtype=np.uint8)
        #cv2.imwrite(f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/results/results_new_synthetic_fog_kitti/{name}.jpg", cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/results/bdd/{f}.jpg", cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB))
        break
