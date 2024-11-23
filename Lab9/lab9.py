import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',) 

#Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Converting images to sobel
sobelHorizontal = cv2.Sobel(gray_image, cv2.CV_64F,1,0,ksize=5)  # x dir
sobelVertical = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5)  # y dir 

#Converting image to canny
canny = cv2.Canny(img, 200, 300) 

#Blurring images
grayBlurredImage = cv2.GaussianBlur(gray_image,(9, 9),0) #Blurring gray image
originalBlurredImage = cv2.GaussianBlur(img, (5, 5), 0) #Blurring original image

#Setting figure size
plt.figure(figsize=(15, 20))  #Width=10, Height=15 

#Plotting 4 images in window
plt.subplot(4, 2,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 2), plt.imshow(gray_image, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

#Plotting blurred images
plt.subplot(4, 2, 3), plt.imshow(cv2.cvtColor(originalBlurredImage, cv2.COLOR_BGR2RGB))
plt.title('5x5 Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 4), plt.imshow(grayBlurredImage, cmap='gray')
plt.title('9x9 Blur'), plt.xticks([]), plt.yticks([])

#Plotting sobel images
plt.subplot(4, 2,5),plt.imshow(sobelHorizontal, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 6), plt.imshow(sobelVertical, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.subplot(4 , 2, 7), plt.imshow(canny, cmap='gray')
plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])

plt.show() 


