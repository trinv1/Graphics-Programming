import cv2 
import numpy as np 
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers

img = cv2.imread('ATU.jpg',) 
img2 = cv2.imread('House.jpg',) 

#Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sobelHorizontal2 = cv2.Sobel(gray_image2, cv2.CV_64F,1,0,ksize=29)  # x dir
sobelVertical2 = cv2.Sobel(gray_image2,cv2.CV_64F,0,1,ksize=29)  # y dir 
sobelSum = sobelHorizontal2 + sobelVertical2
sobelSum2 = sobelHorizontal2 + sobelVertical2

#Converting image to canny
canny2 = cv2.Canny(img2, 100, 200) 

#Blurring images
grayBlurredImage = cv2.GaussianBlur(gray_image,(17, 17),0) #Blurring gray image
originalBlurredImage = cv2.GaussianBlur(img, (11, 11), 0) #Blurring original image

#Normalizing sobelSum to 8-bit range
sobelSum2 = np.abs(sobelSum2)  #Ensuring non-negative values
sobelSum2 = np.uint8(255 * sobelSum2 / np.max(sobelSum2))  #Scaling to 0-255
threshold = np.zeros_like(sobelSum2, dtype=np.uint8)  #Initializing output array for threshold 1
threshold2 = np.zeros_like(sobelSum2, dtype=np.uint8)  #Initializing output array for threshold 2

#Initializing threshold
sobelThreshold1 = 10
sobelThreshold2 = 100

rows, cols = sobelSum2.shape #Getting dimensions of SobelSum image

#Looping through each pixel in sobelSum image
for i in range(rows):
     for j in range(cols):
        if sobelSum2[i, j] > sobelThreshold1:
            threshold[i, j] = 1  # Set to 1 for edges
        else:
            threshold[i, j] = 0  # Set to 0 for background
        
        if sobelSum2[i, j] > sobelThreshold2:
            threshold2[i, j] = 1  
        else:
            threshold2[i, j] = 0  

threshold_visual = threshold * 255  #Scaling binary values to 0 and 255 for display
threshold_visual2 = threshold2 * 255 

#Preparing training data
#Normalizing the grayscale image to [0, 1]
train_input = gray_image2 / 255.0  #Inputting image
train_input = train_input.reshape(1, train_input.shape[0], train_input.shape[1], 1)  #Adding batch and channel dimensions

#Preparing the training label 
train_label = threshold_visual / 255.0  # Normalize the thresholded image to range [0, 1]
train_label = train_label.reshape(1, train_label.shape[0], train_label.shape[1], 1)  # Reshape to match CNN output

print("Shape of train_input:", train_input.shape)
print("Shape of train_label:", train_label.shape)

#Defining the CNN model
model = models.Sequential([
    layers.Conv2D(1, (3, 3), activation='relu', padding='same',input_shape=(None, None, 1)),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Second convolution layer to refine the edges
])

#Compiling the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=losses.MeanSquaredError(),
              metrics=['mse'])

#Training the model
history = model.fit(train_input, train_label, epochs=10, batch_size=1, verbose=1)

#Testing the model on the input image
predicted_output = model.predict(train_input)

#Scaling the output back to 0-255 for visualization
predicted_output_visual = (predicted_output[0, :, :, 0] * 255).astype(np.uint8)

#Setting figure size
plt.figure(figsize=(16, 12))  #Width=10, Height=15 

#Plotting images
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

#Plotting images
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

plt.subplot(4, 2, 5), plt.imshow(sobelSum, cmap='gray')
plt.title('Sobel Sum'), plt.xticks([]), plt.yticks([])

#Plotting canny image
plt.subplot(4, 2, 6), plt.imshow(canny2, cmap='gray')
plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show() 

#Plotting thresholded images
plt.subplot(2, 2, 1), plt.imshow(threshold * 255, cmap='gray')
plt.title('Thresholded 10'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(threshold2 * 255, cmap='gray')
plt.title('Thresholded 100'), plt.xticks([]), plt.yticks([])

#Plotting
plt.subplot(2, 2, 3)
plt.imshow(predicted_output_visual, cmap='gray')
plt.title('CNN Edge Detection')

plt.tight_layout()  # Automatically adjusts subplot spacing

plt.show() 


