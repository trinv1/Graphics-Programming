import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('ATU1.jpg',) 

#Converting image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgHarris = img.copy()

#Performing corner detection 
dst = cv2.cornerHarris(gray_image, 2, 3, k = 0.04) 

#Setting threashold for marking corners
threshold = 0.3; #number between 0 and 1 
for i in range(len(dst)): 
    for j in range(len(dst[i])): 
        if dst[i][j] > (threshold*dst.max()): 
            cv2.circle(imgHarris,(j,i),3,(0, 255, 0),-1) 

plt.figure(figsize=(26, 24))  #Width=10, Height=15 

#Plotting grayscale image
plt.subplot(2, 2, 1), plt.imshow(gray_image, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

#Plotting image copy
plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Image copy'), plt.xticks([]), plt.yticks([])

plt.tight_layout()  # Automatically adjusts subplot spacing
plt.show() 