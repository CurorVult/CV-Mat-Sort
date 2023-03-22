import cv2
import numpy as np

capture = cv2.VideoCapture(0)
# Get the actual frame dimensions of the webcam
widthImg = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
heightImg = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set Camera Parameters
capture.set(10, 130)
IPKernel = np.ones((5,5))

# Use the calculated frame dimensions
capture.set(3, widthImg)
capture.set(4, heightImg)


def get_frame():
    success, img = capture.read()
    img = cv2.resize(img, (widthImg, heightImg))
    return success, img