import cv2
import glob
import numpy as np
import os
import sys
import time
import gzip
import shutil
# Allow accesing files relative to this file
location = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(location, '../../'))
from ocr.viz import print_progress_bar


def extract(location, output, number=5):
    output = os.path.join(location, output)
    if not os.path.exists(output):
        os.makedirs(output)

    for sub in ['lob', 'numbers']:
        folder = os.path.join(location, sub)
        seg_files = glob.glob(os.path.join(folder, '*.seg'))
        length = sum([int(open(l, 'r').readline()) for l in seg_files])

        itr = 0
        for fl in seg_files:
            # Uncompressing tiff files
            with gzip.open(fl[:-4] + '.tiff.gz', 'rb') as f_in:
                with open(fl[:-4] + '.tiff', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
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
                                output,
                                '%s_%s_%s.png' % (word, number, time.time())),
                            im)
                    print_progress_bar(itr, length)
                    itr += 1

    print("\tNumber of words:", len([n for n in os.listdir(output)]))
