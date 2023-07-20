
# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure each string is either "ODD"
# or "EVEN" (without the quotes) depending on whether the hexadecimal number in
# the image is odd or even. Take care not to make spelling or case mistakes. Make
# sure that the length of the list returned as output is the same as the number of
# filenames that were given as input. The judge may give unexpected results if this
# convention is not followed.

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys
import colorsys
import time

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## Defining the shifting function that shifts image to top left corner

def shift_image(img):

    height, width = img.shape[0], img.shape[1]

    blank_image = np.zeros((height, width,3), np.uint8)		## Creating a blank white image
    blank_image.fill(255)

    h1,w1 = 0,0
    found_black_pixel = False

    for i in range(height):									## height from which black pixels start
        for j in range(width):
            if not np.array_equal(img[i,j], np.array([255, 255, 255])):
                h1 = i
                found_black_pixel = True
                break
        if found_black_pixel == True:
            break

    found_black_pixel = False

    for j in range(width):									## width from which black pixels start
        for i in range(height):
            if not np.array_equal(img[i,j], np.array([255, 255, 255])):
                w1 = j
                found_black_pixel = True
                break
        if found_black_pixel == True:
            break

    # print(h1,w1)                  						## For testing purposes

    blank_image[0:height-h1, 0:width-w1] = img[h1:height, w1:width]		## Pasting the image to the top left corner of the blank image

    return blank_image


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## Defining a function that isolates the grayscale image of the last digit of the captcha  and shifts it to the top left corner

def last_digit(img):
    cropped_img = img[0:100, 350:450]                       ## Isolating the last digit by cropping the captcha image

    ## Removing background colors, obfuscating lines which have 'value' of HSV between 0.8 and 1.0
    for i in range(100):
        for j in range(100):
            val = colorsys.rgb_to_hsv(cropped_img[i, j][2] / 255.0, cropped_img[i, j][1] / 255.0, cropped_img[i, j][0] / 255.0)[2]
            if val >= 0.8 and val <= 1.0:
                cropped_img[i, j] = [255, 255, 255]  ## Painting them white
            else:
                cropped_img[i, j] = [0, 0, 0]  ## Painting the remnant black

    final_img = shift_image(cropped_img)                    ## Shifting the digit to the top-left corner

    ## Return the edited image
    return final_img



angles = [30, 20, 10, 0, -10, -20, -30]                     ## These are the rotation angles

num_vect = 7*16     										## Number of images in our training set
vect_len = 100*100*3

train_labels = np.zeros(num_vect)									## Creating the labels for our training data
hex_array = [0,1,2,3,4,5,6,7,8,9,'A','B','C','D','E','F']
for z in range(16):
  for i in range(7):
    train_labels[z*7 + i] = z


Y_train = train_labels

## Creating the training vector

X_train = np.empty((num_vect, vect_len))            ## Declaring an empty vector that will get filled up

for i in range(num_vect):
    num = i%7
    img = cv2.imread('ref_rotated/'+str(hex_array[(int(i/7))])+'_'+str(angles[num])+'.png')
    feature_vector = img.flatten()

    X_train[i] = np.array([feature_vector])

## Training the K-Neearest Neighbours classifier

k = 1  # Number of neighbors to consider for this algorithm
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, Y_train)


def decaptcha( filenames ):									## Actually these are filepaths
    # Invoke your model here to make predictions on the images

    labels = []

    for file in filenames:

        test_image = cv2.imread(file)

        ## Creating the feature vector of the image
        test_feature_vector = ((last_digit(test_image))).flatten()

        ## Comparing the test image with the training images using k-NN
        predicted_labels = knn.predict(np.array([test_feature_vector]))

        ## Next we determine the predicted label
        predicted_label = predicted_labels[0]

        if int(predicted_label)%2 == 0:
            labels.append('EVEN')               ## Even hexadecimal number
        else:
            labels.append('ODD')                ## Odd hexadecimal number

    return labels









"""


## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## This code was used for creating the rotated versions of the 16 hexadecimal images

## Defining the rotation angles
angles = [30, 20, 10, 0, -10, -20, -30]

for z in hex_array:

    ## Loading the image and getting it's height and width
    image = cv2.imread('reference/'+str(z)+'.png')
    height, width = img.shape[0], img.shape[1]

    for angle in angles:

        center = (width/2, height/2)
        rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        ## Applying rotation
        rotated_image = cv2.warpAffine(image, rotate_matrix, (width, height), borderValue=(255, 255, 255))

        cv2.imwrite('ref_rotated/'+str(z)+'_'+str(angle)+'.png', shift_image(rotated_image) )

# Reference:
# https://www.geeksforgeeks.org/python-opencv-getrotationmatrix2d-function/



"""