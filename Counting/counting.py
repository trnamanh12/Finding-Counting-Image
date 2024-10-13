import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('rabbit2.png')

cv2.imshow('image', image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image', gray_image)

_, threshold_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
# cv2.threshold(image, threshold_value, max_value, threshold_type)
# cv2.threshold returns two values, first value is the threshold value and second value is the threshold image
# cv2.THRESH_BINARY_INV is the threshold type which is used to convert the image to binary image
cv2.imshow('threshold_image', threshold_image)

kernel = np.ones((5, 5), np.uint8)
# kernel np.ones((5,5)) because we want to apply the morphological operation on the image
# 
clean_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)
# cv2.morphologyEx(image, operation, kernel) is used to perform morphological operations on the image
# cv2.MORPH_OPEN is the operation which is used to remove noise or small object from the image

cv2.imshow('clean_image', clean_image)

contours, _ = cv2.findContours(clean_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.findContours(image, mode, method) is used to find the contours in the image
# cv2.RETR_EXTERNAL is the mode which retrieves only the external contours, external contours are the contours that are on the edge of the image, internal contours are the contours that are inside the image
# cv2.CHAIN_APPROX_SIMPLE is the method which compresses the contours by removing redundant points

rabbit_count = 0
for contour in contours:
    area = cv2.contourArea(contour)
    # cv2.contourArea(contour) is used to calculate the area of the contour
    # it work by calculating the area of the bounding rectangle of the contour
    
    if area > 500:  # Filter out very small areas that are unlikely to be rabbits
        rabbit_count += 1

output_image = image.copy()
cv2.drawContours(output_image, contours, -1, (0, 0, 0), 2)
# cv2.drawContours(image, contours, contour_index, color, thickness) is used to draw the contours on the image
cv2.imshow("Output Image", output_image)
cv2.waitKey(0) # wait for any key to be pressed
cv2.destroyAllWindows()  # close all windows

print(f"Number of object detected: {rabbit_count}")

