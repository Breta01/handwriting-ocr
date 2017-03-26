# -*- coding: utf-8 -*-
"""
Helper functions for ocr project
"""
import matplotlib.pyplot as plt
import cv2

SMALL_HEIGHT = 800

def implt(img, cmp=None, t=''):
    """ Show image using plt """
    plt.imshow(img, cmap=cmp)
    plt.title(t)
    plt.show()


def resize(img, height=SMALL_HEIGHT, allways=False):
    """ Resize image to given height """
    if (img.shape[0] > height or allways):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))
    
    return img


def ratio(img, height=SMALL_HEIGHT):
    """ Getting scale ratio """
    return img.shape[0] / height