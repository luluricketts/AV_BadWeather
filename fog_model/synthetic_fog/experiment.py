import cv2
import math
import numpy as np
import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pylab as plt
import os

name ="000015_10"
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    # dc=dc+0.2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)
    indices = darkvec.argsort()
    d_nos = imvec[indices[:-1000]]
    A = np.mean(d_nos)
    # print(A*255)
    # indices = indices[imsz-numpx::]
    # atmsum = np.zeros([1,3])
    # for ind in range(1,numpx):
    #    atmsum = atmsum + imvec[indices[ind]]
    # A = atmsum / numpx
    return A*255

def thresh(img):
    im1 = rgb2lab(img)
    return im1
    
path = f"/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/gt"
for file in os.listdir(path):
    filename, ext = os.path.splitext(file)
    path_img = f"/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/gt/{filename}.jpg"
    I = cv2.imread(path_img)
    I = I.astype('float32')#/255
    img = rgb2lab(I)

    cv2.imwrite(f"/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/gt/gray/{filename}_gray.jpg", img)
# dark = DarkChannel(I,2)
# dark1 = dark*255
    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print(grayscaled)
# print(np.max(grayscaled))
# print(np.min(grayscaled))

    ret,thresh1 = cv2.threshold(grayscaled, 200, 9341.572, cv2.THRESH_BINARY)

# cv2.imwrite(f"/home/roshini/AV_BadWeather/TransWeather/synthetic_fog/sample/bdd/night/fe0e0298-7477653e_dark.jpg", dark1)
    cv2.imwrite(f"/home/roshini/AV_BadWeather/TransWeather/data/bdd_night/test/gt/thresh/{filename}_thresh.jpg", thresh1)

# A = AtmLight(I,dark)
# print(A)