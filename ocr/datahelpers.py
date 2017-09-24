# -*- coding: utf-8 -*-
"""
Helper functions for loading and creating datasets
"""
import numpy as np
import glob
import simplejson
import cv2
from .helpers import implt
from .normalization import letterNorm


CHARS_CZ = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
            'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
            'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
            'x', 'y', 'z', 'Á', 'É', 'Í', 'Ó', 'Ú', 'Ý', 'á',
            'é', 'í', 'ó', 'ú', 'ý', 'Č', 'č', 'Ď', 'ď', 'Ě',
            'ě', 'Ň', 'ň', 'Ř', 'ř', 'Š', 'š', 'Ť', 'ť', 'Ů',
            'ů', 'Ž', 'ž']

CHARS_EN = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
            'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
            'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
            'x', 'y', 'z']

idxs = [i for i in range(len(CHARS_CZ))]
idx_to_chars_cz = dict(zip(idxs, CHARS_CZ))
chars_to_idx_cz = dict(zip(CHARS_CZ, idxs))

idxs = [i for i in range(len(CHARS_EN))]
idx_to_chars_en = dict(zip(idxs, CHARS_EN))
chars_to_idx_en = dict(zip(CHARS_EN, idxs))

def char2idx(c, lang='cz'):
    if lang == 'en':
        return chars_to_idx_en[c]
    return chars_to_idx_cz[c]

def idx2char(idx, lang='cz'):
    if lang == 'en':
        return idx_to_chars_en[idx]
    return idx_to_chars_cz[idx]
    

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
    print("-> Number of words:", len(labels))

    # Print one of the images (last one)
    if debug:
        implt(images[-1], 'gray', 'Example')
        print("Word:", labels[-1])
        if loadGaplines:
            print("Gaplines:", gaplines[-1])
    
    if loadGaplines:
        return (images, labels, gaplines)
    return (images, labels)


def words2chars(images, labels, gaplines):
    """ Transform word images with gaplines into individual chars """
    # Total number of chars
    length = sum([len(l) for l in labels])
    
    imgs = np.empty(length, dtype=object)
    newLabels = []
    
    height = images[0].shape[0]
    
    idx = 0;
    for i, gaps in enumerate(gaplines):
        for pos in range(len(gaps) - 1):
            imgs[idx] = images[i][0:height, gaps[pos]:gaps[pos+1]]
            newLabels.append(char2idx(labels[i][pos]))
            idx += 1
           
    print("Loaded chars from words:", length)            
    return imgs, newLabels


def loadCharsData(charloc='data/charclas/', wordloc='data/words/', lang='cz', useWords=True):
    """
    Load chars images with corresponding labels
    Input:
        charloc      - char images FOLDER LOCATION
        wordloc      - word images with gaplines FOLDER LOCATION
    Returns: (images, labels)
    """
    print("Loading chars...")
    # Get subfolders with chars
    dirlist = glob.glob(charloc + lang + "/*/")
    dirlist.sort()

    if lang == 'en':
        chars = CHARS_EN
    else:
        chars = CHARS_CZ
    
    images = np.zeros((1, 4096))
    labels = []   

    # For every label load images and create corresponding labels
    # cv2.imread(img, 0) - for loading images in grayscale
    # Images are scaled to 64x64 = 4096 px
    for i in range(len(chars)):
        imglist = glob.glob(dirlist[i] + '*.jpg')
        imgs = np.array([letterNorm(cv2.imread(img, 0)) for img in imglist])
        images = np.concatenate([images, imgs.reshape(len(imgs), 4096)])
        labels.extend([i] * len(imgs))
        
    if useWords:    
        imgs, words, gaplines = loadWordsData(wordloc)
        imgs, chars = words2chars(imgs, words, gaplines)
        
        labels.extend(chars)
        for i in range(len(imgs)):
            images = np.concatenate([images,
                                     letterNorm(imgs[i]).reshape(1, 4096)])            

    images = images[1:]
    labels = np.array(labels)
    
    print("-> Number of chars:", len(labels))
    return (images, labels)


def correspondingShuffle(a, b):
    """ 
    Shuffle two numpy arrays such that
    each pair a[i] and b[i] remains the same
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]