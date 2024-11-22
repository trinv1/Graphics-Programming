import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',) 

#Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Blurring images
grayBlurredImage = cv2.GaussianBlur(gray_image,(9, 9),0) #Blurring gray image
originalBlurredImage = cv2.GaussianBlur(img, (5, 5), 0) #Blurring original image

#Plotting 4 images in window
plt.subplot(2, 2,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(gray_image, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(originalBlurredImage, cv2.COLOR_BGR2RGB))
plt.title('5x5 Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(grayBlurredImage, cmap='gray')
plt.title('9x9 Blur'), plt.xticks([]), plt.yticks([])

plt.show() 

#Displaying blurred imaged in opencv
cv2.imshow('9x9 Blur', grayBlurredImage)
cv2.imshow('3x3 Blur', originalBlurredImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

