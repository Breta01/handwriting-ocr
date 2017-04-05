# -*- coding: utf-8 -*-
"""
Include functions for normalizing images of words and letters
Main functions: imageNorm and letterNorm
"""
import numpy as np
import cv2
from .helpers import *


def wordTilt(img, height):
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
    return cropAddBorder(img, height, 50)

        
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


def imageNorm(image, height):
    """ 
    Preprocess image
    => resize, get edges, tilt world
    """
    image = resize(image, height, True)
    img = cv2.bilateralFilter(image, 0, 30, 30)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 
    edges = sobelDetect(gray)
    ret,th = cv2.threshold(edges, 50, 255, cv2.THRESH_TOZERO)
    return wordTilt(th, height)


def cropAddBorder(img, height, threshold=0):
    """ Crop and add border to word image of letter segmentation """
    # Clear small values
    ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)
    # Mask of pixels brighter than threshold
    mask = img > threshold
    coords = np.argwhere(mask)
    try:
        # Bounding box of non-black pixels.
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        # Croping image
        resize(img[x0:x1, y0:y1], height, True)
    except Exception:
        pass
    return cv2.copyMakeBorder(img, 0, 0, 15, 15,
                              cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])


def resizeLetter(img, size = 56):
    """ Resize bigger side of the image to given size """
    if (img.shape[0] > img.shape[1]):
        rat = size / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), size))
    else:
        rat = size / img.shape[1]
        return cv2.resize(img, (size, int(rat * img.shape[0])))
    return img


def autocrop(image, threshold=80):
    """ Crops edges below or equal to threshold """
    rows = np.where(np.max(image, 0) > threshold)[0]
    cols = np.where(np.max(image, 1) > threshold)[0]
    image = image[cols[0]:cols[-1] + 1, rows[0]:rows[-1] + 1]
    return image


def letterNorm(image):
    """ Preprocess an image - crop """
    image = autocrop(image)
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
    return result