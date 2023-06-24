import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im
from scipy.ndimage import gaussian_filter
import os
path = f"/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/gt"
for file in os.listdir(path):
    filename, ext = os.path.splitext(file)
    # filename='5bf1efda-9e4be9fe'
    img = plt.imread(os.path.join(path, file))
    # depth_img_path = f"/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/gt/depth/{filename}_depth.jpg"
    depth_path = f"/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/gt/depth/{filename}_depth.npy"
    depth_np = np.load(depth_path)
    depth_img = depth_np[0][0]
    depth_img = np.where(depth_img < depth_img.max()+depth_img.min(), depth_img, depth_img)
    # depth_img = gaussian_filter(depth_img, sigma = 40)
    data = im.fromarray(depth_img)
    if data.mode != 'L':
        data = data.convert('L')
    data.save(f'/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/gt/depth/{filename}_depth.jpg')
    depth_img_path = f'/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/gt/depth/{filename}_depth.jpg'
    depth_img = plt.imread(depth_img_path)
    depth_img = gaussian_filter(depth_img, sigma = 30)
    depth_img_3c = np.zeros_like(img)
    depth_img_3c[:,:,0] = depth_img
    depth_img_3c[:,:,1] = depth_img
    depth_img_3c[:,:,2] = depth_img
    norm_depth_img = depth_img_3c/255
    beta = 5.0
    A= 200
    thresh_path = f"/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/gt/thresh/{filename}_thresh.jpg"
    A_img = plt.imread(thresh_path)
    A_img = np.where(A_img!=0, 1, A_img)*A
    A_img = gaussian_filter(A_img, sigma = 50)
    cv2.imwrite(f"/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/gt/blur_gray/{filename}_result.jpg", cv2.cvtColor(A_img, cv2.COLOR_BGR2RGB))
    A_img_3c = np.zeros_like(img)
    A_img_3c[:,:,0] = A_img
    A_img_3c[:,:,1] = A_img
    A_img_3c[:,:,2] = A_img
    trans = np.exp(-norm_depth_img*beta)
    hazy = (img*trans + A_img_3c*(1-trans))
    hazy = np.array(hazy, dtype=np.uint8)
    cv2.imwrite(f"/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/input/{filename}_result.jpg", cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB))
