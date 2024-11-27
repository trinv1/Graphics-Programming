import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('ATU1.jpg',) 
img2 = cv2.imread('ATU2.jpg',) 

#Converting image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#Copying original image
imgHarris = img.copy()
imgShiTomasi = img.copy()
imgORB = img.copy()

imgHarris2 = img2.copy()
imgShiTomasi2 = img2.copy()
imgORB2 = img2.copy()

#Performing corner detection 
dst = cv2.cornerHarris(gray_image, 2, 3, k = 0.04)
dst2 = cv2.cornerHarris(gray_image2, 2, 3, k = 0.04) 

#Setting threashold for marking corners
threshold = 0.3; #number between 0 and 1 
for i in range(len(dst)): 
    for j in range(len(dst[i])): 
        if dst[i][j] > (threshold*dst.max()): 
            cv2.circle(imgHarris,(j,i),3,(0, 255, 0),-1) 

#Setting threashold for marking corners
threshold2 = 0.3; #number between 0 and 1 
for i in range(len(dst2)): 
    for j in range(len(dst2[i])): 
        if dst2[i][j] > (threshold2*dst2.max()): 
            cv2.circle(imgHarris2,(j,i),3,(0, 255, 0),-1)

#Doing Shi Tomasi corner detection
corners = cv2.goodFeaturesToTrack(gray_image, 40, 0.01, 10)
corners = np.int0(corners) 

corners2 = cv2.goodFeaturesToTrack(gray_image2, 40, 0.01, 10)
corners2 = np.int0(corners2) 

#Looping through corners
for i in corners: 
    x,y = i.ravel() 
    cv2.circle(imgShiTomasi,(x,y),3,(0, 255, 0),-1) 

#Looping through corners
for i in corners2: 
    x,y = i.ravel() 
    cv2.circle(imgShiTomasi2,(x,y),3,(0, 255, 0),-1)

plt.figure(figsize=(10, 15))  #Width=10, Height=15 

#Initiating ORB detector
orb = cv2.ORB_create()
orb2 = cv2.ORB_create()
 
#Finding the keypoints with ORB
kp = orb.detect(imgORB,None)
kp2 = orb2.detect(imgORB2,None)
 
#Computing the descriptors with ORB
kp, des = orb.compute(imgORB, kp)
kp2, des2 = orb2.compute(imgORB2, kp2)

#Plotting grayscale image
plt.subplot(2, 2, 1), plt.imshow(gray_image, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

#Plotting image copy Harris
plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris Image'), plt.xticks([]), plt.yticks([])

#Plotting image copy Shi Tomasi
plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB))
plt.title('Shi Tomasi Image'), plt.xticks([]), plt.yticks([])

#Plotting image ORB
img2 = cv2.drawKeypoints(imgORB, kp, None, color=(0,255,0), flags=0)
plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('ORB Image'), plt.xticks([]), plt.yticks([])

plt.tight_layout()  # Automatically adjusts subplot spacing
plt.show() 

#Plotting grayscale image
plt.subplot(2, 2, 1), plt.imshow(gray_image2, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

#Plotting image copy Harris
plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(imgHarris2, cv2.COLOR_BGR2RGB))
plt.title('Harris Image'), plt.xticks([]), plt.yticks([])

#Plotting image copy Shi Tomasi
plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(imgShiTomasi2, cv2.COLOR_BGR2RGB))
plt.title('Shi Tomasi Image'), plt.xticks([]), plt.yticks([])

#Plotting image ORB
img3 = cv2.drawKeypoints(imgORB2, kp2, None, color=(0,255,0), flags=0)
plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.title('ORB Image'), plt.xticks([]), plt.yticks([])

plt.tight_layout()  # Automatically adjusts subplot spacing
plt.show() 
