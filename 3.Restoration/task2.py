from IPython.display import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as ex
from scipy import signal
from roipoly import RoiPoly

import bsfilter
import checkerboard
import deconvwnr
import imnoise2
import spfilt

pic1_orig = cv2.imread('pic1.jpg',0)
image1_np=np.array(pic1_orig)

deviation = np.std(image1_np)
mean = np.mean(image1_np)
noise = imnoise2.imnoise2('gaussian', (256,256), mean, deviation)

plt.figure()
plt.subplot(3,2,1), plt.xticks([]), plt.yticks([]), plt.imshow(pic1_orig, cmap='gray')
my_roi = RoiPoly(color='r')
plt.subplot(3,2,2), plt.hist(pic1_orig.ravel(), bins=256, range=(0,255))
plt.subplot(3,2,3), plt.hist(noise.ravel(), bins=256, range=(0,255))
plt.show()

print ("\n ----- Standard deviation = " + str(deviation))
print ("\n ----- The population mean = " + str(mean))