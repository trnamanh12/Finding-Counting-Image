import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = "1org.png"
image = cv2.imread(image_path)
# 
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.imshow(image_rgb)
# plt.axis('off')
# plt.show()
# cv2.imshow('image', image)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
# cv2.GaussianBlur(image, kernel_size, sigma_x) is used to apply Gaussian Blur to the image
# sigma_x is the standard deviation in the x-direction
# standard deviation is a measure of the amount of variation or dispersion of a set of values

# Perform edge detection using Canny Edge Detection
edges = cv2.Canny(blurred_image, 50, 120)
# cv2.Canny(image, threshold1, threshold2) is used to perform edge detection using the Canny algorithm
# threshold1 and threshold2 are the minimum and maximum intensity gradient values
# gradient is the rate of change of intensity in the image

# Display the edge-detected image
# plt.imshow(edges, cmap='gray')
# plt.axis('off')
# plt.show()

# Load a template of the object you want to find (e.g., ice cream)
template_path = 'icecream.png'
# process template image

template = cv2.imread(template_path, 0)

template = cv2.Canny(template, 50, 120)
# Perform template matching
result = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
# cv2.matchTemplate(image, template, method) is used to perform template matching
# cv2.TM_CCOEFF_NORMED is the correlation coefficient normalized method
# This method compares the similarity between the template and the image at each location

# Set a threshold to find the object
threshold = 0.088
locations = cv2.minMaxLoc(result)
# cv2.minMaxLoc() returns the minimum and maximum intensity values and their locations in the image
loc = np.where(result >= threshold)
print(loc)
# Get the location of the best match
min_val, max_val, min_loc, max_loc = locations
print(max_val)

# Draw a rectangle around the detected object in the original image
h, w = template.shape
print(template.shape)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image_rgb, top_left, bottom_right, (0, 0, 0), 2)

# for pt in zip(*loc[::-1]): # loc[::-1] is the list of coordinates where the template is found
#     tl = pt
#     bt = (tl[0] + w, tl[1] + h)
#     cv2.rectangle(image_rgb, tl , bt , (0,0, 0), 2) # (0, 255, 0) is the color of the rectangle
    
    # cv2.rectangle(image, top_left, bottom_right, color, thickness) is used to draw a rectangle around the object
    

# Display the result

# cv2.rectangle(image, top_left, bottom_right, color, thickness) is used to draw a rectangle around the object

# show the image
cv2.imshow('image', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Display the result
# plt.imshow(image_rgb)
# plt.axis('off')
# plt.show()
