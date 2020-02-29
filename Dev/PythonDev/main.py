# import cv2
# import numpy as np
#
# # image = cv2.imread('img24.jpg')
# # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# # sharpened = cv2.filter2D(image, -1, kernel)
# # #sharpened = cv2.filter2D(image, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
# # cv2.imshow('Image Sharpening', sharpened)
# # cv2.imwrite('img24improved.jpg',sharpened)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
###################################################################################################################################
# # importing required libraries of opencv
# import cv2
#
# # importing library for plotting
# from matplotlib import pyplot as plt
#
# # reads an input image
# img = cv2.imread('Work/img2improved.jpg', 0)
#
# # find frequency of pixels in range 0-255
# histr = cv2.calcHist([img], [0], None, [256], [0, 256])
#
# # show the plotting graph of an image
# plt.plot(histr)
# plt.show()
##################################################################################
from skimage.measure import compare_ssim
import cv2
import numpy as np

before = cv2.imread('Work/img9.jpg')
after = cv2.imread('Work/img9improved.jpg')
height, width, depth = before.shape
after = cv2.resize(after,(width,height))   # image resizing


# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = compare_ssim(before_gray, after_gray, full=True)
print("Image similarity", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

# cv2.imshow('before', before)
# cv2.imshow('after', after)
# cv2.imshow('diff',diff)
# cv2.imshow('mask',mask)
cv2.imwrite('diff.jpg',diff)
cv2.imshow('filled after',filled_after)
cv2.waitKey(0)
########################################################################################################################
# from skimage.measure import compare_ssim
# import cv2
# import numpy as np

# image = cv2.imread('Work/img9.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('diff.jpg',gray)