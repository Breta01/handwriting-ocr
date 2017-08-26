# -*- coding: utf-8 -*-
"""
Helper functions for loading and creating datasets
"""
import numpy as np
import glob
import simplejson
import cv2
from .helpers import implt

def loadWordsData(dataloc='data/words/', loadGaplines=True, debug=False):
    """
    Load word images with corresponding labels and gaplines (if loadGaplines == True)
    Input:
        dataloc      - image folder location - can be list of multiple locations,
        loadGaplines - wheter or not load gaplines positions files
        debug        - for printing example image
    Returns: (images, labels (, gaplines))
    """
    print("Loading words...")
    imglist = []
    tmpLabels = []
    if type(dataloc) is list:
        for loc in dataloc:
            tmpList = glob.glob(loc + '*.jpg')
            imglist += tmpList
            tmpLabels += [name[len(loc):].split("_")[0] for name in tmpList]
    else:
        imglist = glob.glob(dataloc + '*.jpg')
        tmpLabels = [name[len(dataloc):].split("_")[0] for name in imglist]
    
    labels = np.array(tmpLabels)
    images = np.empty(len(imglist), dtype=object)

    # Load grayscaled images
    for i, img in enumerate(imglist):
        images[i] = cv2.imread(img, 0)
    
    # Load gaplines (lines separating letters) from txt files
    if loadGaplines:
        gaplines = np.empty(len(imglist), dtype=object)
        for i, name in enumerate(imglist):
            with open(name[:-3] + 'txt', 'r') as fp:
                gaplines[i] = np.array(simplejson.load(fp))
                
    # Check the same lenght of labels and images
    if loadGaplines:
        assert len(labels) == len(images) == len(gaplines)
    else:
        assert len(labels) == len(images)
    print("Number of Images:", len(labels))

    # Print one of the images (last one)
    if debug:
        implt(images[-1], 'gray', 'Example')
        print("Word:", labels[-1])
        if loadGaplines:
            print("Gaplines:", gaplines[-1])
    
    if loadGaplines:
        return (images, labels, gaplines)
    return (images, labels)


def correspondingShuffle(a, b):
    """ 
    Shuffle two numpy arrays such that
    each pair a[i] and b[i] remains the same
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]