

import cv2
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from scipy import signal as sig

img = imread('chess.jpg')
imggray = rgb2gray(img)

def gradient_x(imggray):
    ##Sobel operator kernels.
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return sig.convolve2d(imggray, kernel_x, mode='same')
def gradient_y(imggray):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sig.convolve2d(imggray, kernel_y, mode='same')

gx = gradient_x(imggray)
gy = gradient_y(imggray)
'''
gx = cv2.Sobel(img1, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(img1, cv2.CV_32F, 0, 1)
'''
Ixx = gx ** 2
Ixy = gy * gx
Iyy = gy ** 2
k = 0.2

height, width = imggray.shape
harris_response = []
offset=3



for y in range(offset, height - offset):
    for x in range(offset, width - offset):
        Sxx = np.sum(Ixx[y-offset:y + 1 + offset, x - offset:x + 1 + offset])
        Syy = np.sum(Iyy[y-offset:y + 1 + offset, x - offset:x + 1 + offset])
        Sxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
        det = (Sxx * Syy) - (Sxy ** 2)
        trace = Sxx + Syy
        r = det - k * (trace ** 2)
        harris_response.append([x, y, r])

img_copy = np.copy(img)

for response in harris_response:
    x, y, r = response
    if r > 0.01:
        img_copy[y,x] = [255, 0, 0]



cv2.imshow('square-circle-2',img_copy )

cv2.waitKey(0)
