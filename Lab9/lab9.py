import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',) 

#Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Plotting 2 images in window
plt.subplot(1, 2,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
plt.title('Original'), plt.xticks([]), plt.yticks([]) 
plt.subplot(1, 2,2),plt.imshow(gray_image, cmap = 'gray') 
plt.title('GrayScale'), plt.xticks([]), plt.yticks([]) 
plt.show() 

#Blurred image
imgOut = cv2.GaussianBlur(img,(9, 9),0) 
cv2.imshow('Blurred Image', imgOut)
cv2.waitKey(0)
cv2.destroyAllWindows()
