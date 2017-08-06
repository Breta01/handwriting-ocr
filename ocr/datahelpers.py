# -*- coding: utf-8 -*-
"""
Helper functions for loading and creating datasets
"""
import numpy as np
import glob
import simplejson
import cv2
from .helpers import implt

def loadWordsData(dataloc='data/words/', debug=False):
    """ 
    Load word images with corresponding labels and gaplines
    Input: image folder location, debug - for printing example image
    Returns: (images, labels, gaplines)
    """
    print("Loading words...")
    imglist = glob.glob(dataloc + '*.jpg')
    imglist.sort()
    
    labels = np.array([name[len(dataloc):].split("_")[0] for name in imglist])
    images = np.empty(len(imglist), dtype=object)
    gaplines = np.empty(len(imglist), dtype=object)
    
    # Load grayscaled images
    for i, img in enumerate(imglist):
        images[i] = cv2.imread(img, 0)    
    
    # Load gaplines (separating letters) from txt files
    for i, name in enumerate(imglist):
        with open(name[:-3] + 'txt', 'r') as fp:
            gaplines[i] = simplejson.load(fp)

    assert len(labels) == len(images) == len(gaplines) # Check the same lenght of labels and images
    print("Number of Images:", len(labels))

    # Print one of the images (last one)
    if debug:
        implt(images[-1], 'gray', 'Example')
        print("Word:", labels[-1])
        print("Gaplines:", gaplines[-1])
        
    return (images, labels, gaplines)


def correspondingShuffle(a, b):
    """ 
    Shuffle two numpy arrays such that
    each pair a[i] and b[i] remains the same
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]