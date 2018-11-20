import cv2
import glob
import numpy as np
import os
import time

arg_folder1 = './../../data/raw/camb/lob'
arg_folder2 = './../../data/raw/camb/numbers'
arg_output = 'words-final'

datasetNum = 5

def dataset_preprocessing(folder, output):
    if not os.path.exists(output):
        os.makedirs(output)
    seg_files = glob.glob(os.path.join(folder, '*.seg'))
    length = sum([int(open(l, 'r').readline()) for l in seg_files])
    itr = 0

    for fl in seg_files:
        image = cv2.imread(fl[:-4] + ".tiff")
        with open(fl) as f:
            f.readline()
            for line in f:
                rect = [int(val) for val in line.strip().split(' ')[1:]]
                word = line.split(' ')[0].split('_')[0]
                im = image[rect[2]:rect[3], rect[0]:rect[1]]

                if 0 not in im.shape:
                    cv2.imwrite(
                        os.path.join(
                            output, '%s_%s_%s.png' % (word, datasetNum, time.time())),
                        im)

                print('%s / %s' % (itr, length), end='\r')
                itr += 1

    print("Number of words:", len([n for n in os.listdir(output)]))

dataset_preprocessing(arg_folder1, arg_output)
dataset_preprocessing(arg_folder2, arg_output)
