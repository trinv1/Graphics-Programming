import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',) 
img2 = cv2.imread('House.jpg',) 

#Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#Converting images to sobel
sobelHorizontal = cv2.Sobel(gray_image, cv2.CV_64F,1,0,ksize=5)  # x dir
sobelVertical = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5)  # y dir 

sobelHorizontal2 = cv2.Sobel(gray_image2, cv2.CV_64F,1,0,ksize=5)  # x dir
sobelVertical2 = cv2.Sobel(gray_image2,cv2.CV_64F,0,1,ksize=5)  # y dir 
sobelSum = sobelHorizontal + sobelVertical

#Converting image to canny
canny = cv2.Canny(img, 100, 450)
canny2 = cv2.Canny(img2, 100, 450) 

#Blurring images
grayBlurredImage = cv2.GaussianBlur(gray_image,(17, 17),0) #Blurring gray image
originalBlurredImage = cv2.GaussianBlur(img, (11, 11), 0) #Blurring original image

#Setting figure size
plt.figure(figsize=(16, 12))  #Width=10, Height=15 

#Plotting 4 images in window
plt.subplot(4, 2,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 2), plt.imshow(gray_image, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

#Plotting blurred images
plt.subplot(4, 2, 3), plt.imshow(cv2.cvtColor(originalBlurredImage, cv2.COLOR_BGR2RGB))
plt.title('11x11 Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 4), plt.imshow(grayBlurredImage, cmap='gray')
plt.title('17x17 Blur'), plt.xticks([]), plt.yticks([])

plt.tight_layout()  # Automatically adjusts subplot spacing

plt.show() 

#Plotting 4 images in 2nd window

#Setting figure size
plt.figure(figsize=(16, 12))  #Width=10, Height=15 

plt.subplot(4, 2,1),plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 2), plt.imshow(gray_image2, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

#Plotting sobel images
plt.subplot(4, 2,3),plt.imshow(sobelHorizontal2, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(4, 2, 4), plt.imshow(sobelVertical2, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

#Plotting canny image
plt.subplot(4 , 2, 5), plt.imshow(canny2, cmap='gray')
plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])

plt.tight_layout()  # Automatically adjusts subplot spacing

plt.show() 


