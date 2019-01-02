import glob
import os
import sys
from shutil import copyfile
import time
# Allow accesing files relative to this file
location = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(location, '../../'))
from ocr.viz import print_progress_bar


def extract(location, output, number=4):
    output = os.path.join(location, output)
    if not os.path.exists(output):
        os.makedirs(output)

    for sub in ['ORAND-CAR-2014/CAR-A', 'ORAND-CAR-2014/CAR-B']:
        folder = os.path.join(location, sub)
        l_files = glob.glob(os.path.join(folder, '*.txt'))
        length = sum(1 for fl in l_files for line in open(fl))

        itr = 0
        for fl in l_files:
            im_folder = fl[:-6] + 'images'
            with open(fl) as f:
                for line in f:
                    im, word = line.strip().split('\t')
                    impath = os.path.join(im_folder, im)

                    if os.stat(impath).st_size != 0:
                        outpath = os.path.join(
                            output,
                            '%s_%s_%s.png' % (word, number, time.time()))
                        copyfile(impath, outpath)
                    print_progress_bar(itr, length)
                    itr += 1

    print("\tNumber of words:", len([n for n in os.listdir(output)]))
