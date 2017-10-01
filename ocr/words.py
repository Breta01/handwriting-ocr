# -*- coding: utf-8 -*-
"""
Detect words on the page
return array of words' bounding boxes
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from .helpers import *

def detection(image):
    """ Detecting the words bounding boxes """
    # Preprocess image for word detection
    blurred = cv2.GaussianBlur(image, (5, 5), 10)
    edgeImg = edgeDetect(blurred)
    ret, edgeImg = cv2.threshold(edgeImg, 50, 255, cv2.THRESH_BINARY)
    bwImage = cv2.morphologyEx(edgeImg, cv2.MORPH_CLOSE,
                               np.ones((15,15), np.uint8))
    # Return detected bounding boxes
    return textDetect(bwImage, image)


def edgeDetect(im):
    """ 
    Edge detection 
    Sobel operator is applied for each image layer (RGB)
    """
    return np.max(np.array([sobelDetect(im[:,:, 0]),
                            sobelDetect(im[:,:, 1]),
                            sobelDetect(im[:,:, 2])]), axis=0)


def sobelDetect(channel):
    """ Sobel operator """
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)


def textDetect(img, image):
    """ Text detection using contours """
    small = resize(img, 2000)
    
    # Finding contours
    mask = np.zeros(small.shape, np.uint8)
    im2, cnt, hierarchy = cv2.findContours(np.copy(small),
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    index = 0    
    boundingBoxes = np.array([0,0,0,0])
    
    # image for drawing bounding boxes
    small = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
    
    # Go through all contours in top level
    while (index >= 0):
        x,y,w,h = cv2.boundingRect(cnt[index])
        cv2.drawContours(mask, cnt, index, (255, 255, 255), cv2.FILLED)
        maskROI = mask[y:y+h, x:x+w]
        # Ratio of white pixels to area of bounding rectangle
        r = cv2.countNonZero(maskROI) / (w * h)
        
        # Limits for text
        if r > 0.1 and 1600 > w > 10 and 1600 > h > 10:
            cv2.rectangle(small, (x, y),(x+w,y+h), (0, 255, 0), 2)
            boundingBoxes = np.vstack((boundingBoxes,
                                       np.array([x, y, x+w, y+h])))
            
        index = hierarchy[0][index][0]
        
    implt(small, t='Bounding rectangles')
    
    bBoxes = boundingBoxes.dot(ratio(image, 2000)).astype(np.int64)
    return bBoxes[1:]  
    

def textDetectWatershed(thresh):
    """ Text detection using watershed algorithm - NOT IN USE """
    # According to: http://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html
    img = cv2.cvtColor(cv2.imread("data/textdet/%s.jpg" % IMG),
                       cv2.COLOR_BGR2RGB)
    img = resize(img, 3000)
    thresh = resize(thresh, 3000)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,
                                 0.01*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers += 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    markers = cv2.watershed(img, markers)
    implt(markers, t='Markers')
    image = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for mark in np.unique(markers):
        # mark == 0 --> background
        if mark == 0:
            continue

        # Draw it on mask and detect biggest contour
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == mark] = 255

        cnts = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        
        # Draw a bounding rectangle if it contains text
        x,y,w,h = cv2.boundingRect(c)
        cv2.drawContours(mask, c, 0, (255, 255, 255), cv2.FILLED)
        maskROI = mask[y:y+h, x:x+w]
        # Ratio of white pixels to area of bounding rectangle
        r = cv2.countNonZero(maskROI) / (w * h)
        
        # Limits for text
        if r > 0.2 and 2000 > w > 15 and 1500 > h > 15:
            cv2.rectangle(image, (x, y),(x+w,y+h), (0, 255, 0), 2)
        
    implt(image)
