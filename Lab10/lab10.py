import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('ATU1.jpg',) 

#Converting image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Plotting grayscale image
plt.subplot(2, 2, 1), plt.imshow(gray_image, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.tight_layout()  # Automatically adjusts subplot spacing

plt.show() 