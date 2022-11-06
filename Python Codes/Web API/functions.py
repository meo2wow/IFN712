import numpy as np
import cv2
import keras
import tensorflow as tf
# this is used for removing the background
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from win32api import GetSystemMetrics

segmentor = SelfiSegmentation()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FUNCTIONS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def rescaleFrame(frame, scale=0.75):
    """
    This rescales a frame with a given scale
    :@param frame:
    :@param scale:
    :@return:
    """
    # print("Width =", GetSystemMetrics(0))
    # print("Height =", GetSystemMetrics(1))
    scale = (GetSystemMetrics(0) * 0.3) / frame.shape[0]
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def show(name, image):
    """
    A simple function to visualize OpenCV images on screen.
    @param name: a string signifying the imshow() window name
    @param image: NumPy image to show
    """
    # Naming a window which will show the image
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # Using resizeWindow() to fit the screen if necessary
    # cv2.resizeWindow(name, 540, 960)
    # show the image/frame
    cv2.imshow(name, image)
    cv2.waitKey(1)

