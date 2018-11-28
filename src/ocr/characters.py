# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import cv2
import math

from .helpers import *
from .tfhelpers import Model

# Preloading trained model with activation function
# Loading is slow -> prevent multiple loads
print("Loading segmentation models...")
location = os.path.dirname(os.path.abspath(__file__))
CNN_model = Model(
    os.path.join(location, '../../models/gap-clas/CNN-CG'))
CNN_slider = (60, 30)
RNN_model = Model(
    os.path.join(location, '../../models/gap-clas/RNN/Bi-RNN-new'),
    'prediction')
RNN_slider = (60, 60)


def _classify(img, step=2, RNN=False, slider=(60, 60)):
    """Slice the image and return raw output of classifier."""
    length = (img.shape[1] - slider[1]) // 2 + 1
    if RNN:
        input_seq = np.zeros((1, length, slider[0]*slider[1]), dtype=np.float32)
        input_seq[0][:] = [img[:, loc * step: loc * step + slider[1]].flatten()
                           for loc in range(length)]
        pred = RNN_model.eval_feed({'inputs:0': input_seq,
                                    'length:0': [length],
                                    'keep_prob:0': 1})[0]
    else:
        input_seq = np.zeros((length, slider[0]*slider[1]), dtype=np.float32)
        input_seq[:] = [img[:, loc * step: loc * step + slider[1]].flatten()
                        for loc in range(length)]
        pred = CNN_model.run(input_seq)
        
    return pred
    

def segment(img, step=2, RNN=False, debug=False):
    """Take preprocessed image of word and
    returns array of positions separating characters.
    """
    slider = CNN_slider
    if RNN:
        slider = RNN_slider
    
    # Run the classifier
    pred = _classify(img, step=step, RNN=RNN, slider=slider)

    # Finalize the gap positions from raw prediction
    gaps = []
    last_gap = 0
    gap_count = 1
    gap_position_sum = slider[1] / 2
    first_gap = True
    gap_block_first = 0
    gap_block_last = slider[1] / 2

    for i, p in enumerate(pred):
        if p == 1:
            gap_position_sum += i * step + slider[1] / 2
            gap_block_last = i * step + slider[1] / 2
            gap_count += 1
            last_gap = 0
            if gap_block_first == 0:
                gap_block_first = i * step + slider[1] / 2
        else:
            if gap_count != 0 and last_gap >= 1:
                if first_gap:
                    gaps.append(int(gap_block_last))
                    first_gap = False
                else:
                    gaps.append(int(gap_position_sum // gap_count))
                gap_position_sum = 0
                gap_count = 0
            gap_block_first = 0
            last_gap += 1

    # Adding final gap position
    if gap_block_first != 0:
        gaps.append(int(gap_block_first))
    else:
        gap_position_sum += (len(pred) - 1) * 2 + slider[1]/2
        gaps.append(int(gap_position_sum / (gap_count + 1)))
        
    if debug:
        # Drawing lines
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for gap in gaps:
            cv2.line(img,
                     ((int)(gap), 0),
                     ((int)(gap), slider[0]),
                     (0, 255, 0), 1)
        implt(img, t="Separated characters")
        
    return gaps