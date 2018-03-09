# -*- coding: utf-8 -*-
"""
Helper functions for loading and creating datasets
"""
import numpy as np
import glob
import simplejson
import cv2
import unidecode
from .helpers import implt
from .normalization import letterNorm
from .viz import printProgressBar


CHARS = ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
         'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
         'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
         'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
         'x', 'y', 'z', 'Á', 'É', 'Í', 'Ó', 'Ú', 'Ý', 'á',
         'é', 'í', 'ó', 'ú', 'ý', 'Č', 'č', 'Ď', 'ď', 'Ě',
         'ě', 'Ň', 'ň', 'Ř', 'ř', 'Š', 'š', 'Ť', 'ť', 'Ů',
         'ů', 'Ž', 'ž']

idxs = [i for i in range(len(CHARS))]
idx_to_chars = dict(zip(idxs, CHARS))
chars_to_idx = dict(zip(CHARS, idxs))

def char2idx(c, sequence=False):
    if sequence:
        return chars_to_idx[c] + 1
    return chars_to_idx[c]

def idx2char(idx, sequence=False):
    if sequence:
        return idx_to_chars[idx-1]
    return idx_to_chars[idx]
    

def loadWordsData(dataloc='data/words/', loadGaplines=True, debug=False):
    """
    Load word images with corresponding labels and gaplines (if loadGaplines == True)
    Args:
        dataloc: image folder location - can be list of multiple locations,
        loadGaplines: wheter or not load gaplines positions files
        debug: for printing example image
    Returns:
        (images, labels (, gaplines))
    """
    print("Loading words...")
    imglist = []
    tmpLabels = []
    if type(dataloc) is list:
        for loc in dataloc:
            loc += '/' if loc[-1] != '/' else ''
            tmpList = glob.glob(loc + '*.jpg')
            imglist += tmpList
            tmpLabels += [name[len(loc):].split("_")[0] for name in tmpList]
    else:
        dataloc += '/' if dataloc[-1] != '/' else ''
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


def words2chars(images, labels, gaplines, lang='cz'):
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
            if lang == 'cz':
                newLabels.append(char2idx(labels[i][pos]))
            else:
                newLabels.append(char2idx(unidecode.unidecode(labels[i][pos])))
            idx += 1
           
    print("Loaded chars from words:", length)            
    return imgs, newLabels


def loadCharsData(charloc='data/charclas/', wordloc='data/words/', lang='cz'):
    """
    Load chars images with corresponding labels
    Args:
        charloc: char images FOLDER LOCATION
        wordloc: word images with gaplines FOLDER LOCATION
    Returns:
        (images, labels)
    """
    print("Loading chars...")
    images = np.zeros((1, 4096))
    labels = []

    if charloc != '':
        # Get subfolders with chars
        dirlist = glob.glob(charloc + lang + "/*/")
        dirlist.sort()    

        if lang == 'en':
            chars = CHARS_EN
        else:
            chars = CHARS_CZ

        assert [d[-2] if d[-2] != '0' else '' for d in dirlist] == chars

        # For every label load images and create corresponding labels
        # cv2.imread(img, 0) - for loading images in grayscale
        # Images are scaled to 64x64 = 4096 px
        for i in range(len(chars)):
            imglist = glob.glob(dirlist[i] + '*.jpg')
            imgs = np.array([letterNorm(cv2.imread(img, 0)) for img in imglist])
            images = np.concatenate([images, imgs.reshape(len(imgs), 4096)])
            labels.extend([i] * len(imgs))
        
    if wordloc != '':    
        imgs, words, gaplines = loadWordsData(wordloc)
        imgs, chars = words2chars(imgs, words, gaplines, lang)
        
        labels.extend(chars)
        for i in range(len(imgs)):
            printProgressBar(i, len(imgs))
            images = np.concatenate([images,
                                     letterNorm(imgs[i]).reshape(1, 4096)])            

    images = images[1:]
    labels = np.array(labels)
    
    print("-> Number of chars:", len(labels))
    return (images, labels)


def loadGapData(loc='data/gapdet/large/', slider=(60, 120), seq=False, flatten=True):
    """ 
    Load gap data from location with corresponding labels
    Args:
        loc: location of folder with words separated into gap data
             images have to by named as label_timestamp.jpg, label is 0 or 1
        slider: dimensions of of output images
        seq: Store images from one word as a sequence
        flatten: Flatten the output images
    Returns:
        (images, labels)
    """
    print('Loading gap data...')
    loc += '/' if loc[-1] != '/' else ''
    dirlist = glob.glob(loc + "*/")
    dirlist.sort()
    
    if slider[1] > 120:
        # TODO Implement for higher dimmensions
        slider[1] = 120
        
    cut_s = None if (120 - slider[1]) // 2 <= 0 else  (120 - slider[1]) // 2
    cut_e = None if (120 - slider[1]) // 2 <= 0 else -(120 - slider[1]) // 2
    
    if seq:
        images = np.empty(len(dirlist), dtype=object)
        labels = np.empty(len(dirlist), dtype=object)
        
        for i, loc in enumerate(dirlist):
            # TODO Check for empty directories
            imglist = glob.glob(loc + '*.jpg')
            if (len(imglist) != 0):
                imgList = sorted(imglist, key=lambda x: int(x[len(loc):].split("_")[1][:-4]))
                images[i] = np.array([(cv2.imread(img, 0)[:, cut_s:cut_e].flatten() if flatten else
                                       cv2.imread(img, 0)[:, cut_s:cut_e])
                                      for img in imglist])
                labels[i] = np.array([int(name[len(loc):].split("_")[0]) for name in imglist])
        
    else:
        images = np.zeros((1, slider[0]*slider[1]))
        labels = []

        for i in range(len(dirlist)):
            imglist = glob.glob(dirlist[i] + '*.jpg')
            if (len(imglist) != 0):
                imgs = np.array([cv2.imread(img, 0)[:, cut_s:cut_e] for img in imglist])
                images = np.concatenate([images, imgs.reshape(len(imgs), slider[0]*slider[1])])
                labels.extend([int(img[len(dirlist[i])]) for img in imglist])

        images = images[1:]
        labels = np.array(labels)
    
    if seq:
        print("-> Number of words / gaps and letters:",
              len(labels), '/', sum([len(l) for l in labels]))
    else:
        print("-> Number of gaps and letters:", len(labels))
    return (images, labels)    


def correspondingShuffle(a):
    """ 
    Shuffle array of numpy arrays such that
    each pair a[x][i] and a[y][i] remains the same
    Args:
        a: array of same length numpy arrays
    Returns:
        Array a with shuffled numpy arrays
    """
    assert all([len(a[0]) == len(a[i]) for i in range(len(a))])
    p = np.random.permutation(len(a[0]))
    for i in range(len(a)):
        a[i] = a[i][p]
    return a


def sequences_to_sparse(sequences):
    """
    Create a sparse representention of sequences.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)
        
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape