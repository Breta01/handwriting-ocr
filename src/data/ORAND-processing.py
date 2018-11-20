import glob
import os
from shutil import copyfile
import time

arg_folder1 = './../../data/raw/orand/ORAND-CAR-2014/CAR-A'
arg_folder2 = './../../data/raw/orand/ORAND-CAR-2014/CAR-B'
arg_output = 'words-final'

datasetNum = 4

def dataset_preprocessing(folder, output):
    if not os.path.exists(output):
        os.makedirs(output)

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
                        output, '%s_%s_%s.png' % (word, datasetNum, time.time()))
                    copyfile(impath, outpath)

                print('%s / %s' % (itr, length), end='\r')
                itr += 1

    print("Number of words:", len([n for n in os.listdir(output)]))


dataset_preprocessing(arg_folder1, arg_output)
dataset_preprocessing(arg_folder2, arg_output)
