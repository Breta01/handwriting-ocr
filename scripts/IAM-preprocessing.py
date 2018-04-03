import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

from ocr.normalization import imageNorm
from ocr.helpers import implt
from ocr.viz import printProgressBar

# TODO replace with input parsers, check if output folder exists
arg_label_file = 'words.txt'    # Location of words.txt file from IAM dataset
arg_folder = 'words'            # Folder location of IAM dataset
arg_output = 'wordsIAM'         # Output folder location

def datasetPreprocessing(label_file, folder, output):
    if folder[-1] != '/':
        folder += '/'
    if output[-1] != '/':
        output += '/'

    length = len(open(label_file).readlines())

    with open(label_file) as fp:
        print("Processing files:")
        for i, line in enumerate(fp):
            printProgressBar(i, length)
            if line[0] != '#':
                l = line.strip().split(" ")
                if l[1] == 'ok':
                    impath = (folder + l[0].split('-')[0] + '/' 
                             + l[0].split('-')[0] + '-' + l[0].split('-')[1]
                             + '/' + l[0] + '.png')
                    img = cv2 .imread(impath)

                    if img is not None:
                        img = imageNorm(
                            img,
                            60,
                            border=False,
                            tilt=True,
                            hystNorm=True)
                        cv2.imwrite(
                            output + "%s_%s.jpg" % (l[-1], time.time()),
                            img)

datasetPreprocessing(arg_label_file, arg_folder, arg_output)
