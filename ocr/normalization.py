# -*- coding: utf-8 -*-
"""
Include functions for normalizing images of words and letters
Main functions: imageNorm, letterNorm, imageStandardization
"""
import numpy as np
import cv2
import math
from .helpers import *



def imageStandardization(image):
    """ Image standardization same as tf.image.per_image_standardization """
    return (image - np.mean(image)) / max(np.std(image), 1.0/math.sqrt(image.size))


def cropAddBorder(img, height, threshold=50, border=True, borderSize=15):
    """ Crop and add border to word image of letter segmentation """
    # Clear small values
    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)

    x0 = 0
    y0 = 0
    x1 = img.shape[1]
    y1 = img.shape[0]

    for i in range(img.shape[0]):
        if np.count_nonzero(img[i, :]) > 1:
            y0 = i
            break
    for i in reversed(range(img.shape[0])):
        if np.count_nonzero(img[i, :]) > 1:
            y1 = i+1
            break
    for i in range(img.shape[1]):
        if np.count_nonzero(img[:, i]) > 1:
            x0 = i
            break
    for i in reversed(range(img.shape[1])):
        if np.count_nonzero(img[:, i]) > 1:
            x1 = i+1
            break
    
    if height != 0:
        img = resize(img[y0:y1, x0:x1], height, True)
    else:
        img = img[y0:y1, x0:x1]
    
    if border:
        return cv2.copyMakeBorder(img, 0, 0, borderSize, borderSize,
                                  cv2.BORDER_CONSTANT,
                                  value=[0, 0, 0])
    return img


def wordTilt(img, height, border=True, borderSize=15):
    """ Detect the angle for tiltByAngle function """
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 30)
    
    if lines is not None:
        meanAngle = 0
        # Set min number of valid lines (try higher)
        numLines = np.sum(1 for l in lines if l[0][1] < 0.7 or l[0][1] > 2.6)
        if numLines > 1:
            meanAngle = np.mean([l[0][1] for l in lines if l[0][1] < 0.7 or l[0][1] > 2.6])

        # Look for angle with correct value
        if meanAngle != 0 and (meanAngle < 0.7 or meanAngle > 2.6):
            img = tiltByAngle(img, meanAngle, height)
    return cropAddBorder(img, height, 50, border, borderSize)

        
def tiltByAngle(img, angle, height):
    """ Tilt the image by given angle """
    dist = np.tan(angle) * height
    width = len(img[0])
    sPoints = np.float32([[0,0], [0,height], [width,height], [width,0]])
    
    # Dist is positive for angle < 0.7; negative for angle > 2.6
    # Image must be shifed to right
    if dist > 0:
        tPoints = np.float32([[0,0],
                              [dist,height],
                              [width+dist,height],
                              [width,0]])
    else:
        tPoints = np.float32([[-dist,0],
                              [0,height],
                              [width,height],
                              [width-dist,0]])

    M = cv2.getPerspectiveTransform(sPoints, tPoints)
    return cv2.warpPerspective(img, M, (int(width+abs(dist)), height))


def sobelDetect(channel):
    """ The Sobel Operator"""
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    # Combine x, y gradient magnitudes sqrt(x^2 + y^2)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)


def imageNorm(image, height, border=True, tilt=True, borderSize=15):
    """ 
    Preprocess image
    => resize, get edges, tilt world
    """
    image = resize(image, height, True)
    img = cv2.bilateralFilter(image, 0, 30, 30)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 
    edges = sobelDetect(gray)
    ret,th = cv2.threshold(edges, 50, 255, cv2.THRESH_TOZERO)
    if tilt:
        return wordTilt(th, height, border, borderSize)
    return th


def resizeLetter(img, size = 56):
    """ Resize bigger side of the image to given size """
    if (img.shape[0] > img.shape[1]):
        rat = size / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), size))
    else:
        rat = size / img.shape[1]
        return cv2.resize(img, (size, int(rat * img.shape[0])))
    return img


def letterNorm(image, is_thresh=True, dim=False):
    """ Preprocess an image - crop """
    if is_thresh and image.shape[0] > 0 and image.shape[1] > 0:
        image = cropAddBorder(image, height=0, threshold=80, border=False) # threshold=80

    resized = image
    if image.shape[0] > 1 and image.shape[1] > 1:
        resized = resizeLetter(image)
    
    result = np.zeros((64, 64), np.uint8)
    offset = [0, 0]
    # Calculate offset for smaller size
    if image.shape[0] > image.shape[1]:
        offset = [int((result.shape[1] - resized.shape[1])/2), 4]
    else:
        offset = [4, int((result.shape[0] - resized.shape[0])/2)]
    # Replace zeros by image 
    result[offset[1]:offset[1] + resized.shape[0],
           offset[0]:offset[0] + resized.shape[1]] = resized
    
    if dim:
        return result, image.shape    
    return result