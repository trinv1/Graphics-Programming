import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',) 

#Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destoryAllWindows()

