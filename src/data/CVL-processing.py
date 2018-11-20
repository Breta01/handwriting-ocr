import enchant
import glob
from PIL import Image
import os
from shutil import copyfile
import time
import re

arg_folder1 = './../../data/raw/cvl/cvl-database-1-1/testset'
arg_folder2 = './../../data/raw/cvl/cvl-database-1-1/trainset'
arg_output = 'words-final'

datasetNum = 3

def dataset_preprocessing(folder, output):
    if not os.path.exists(output):
        os.makedirs(output)
    d = enchant.Dict('en_US')
    images = glob.glob(os.path.join(folder, 'words', '*', '*.tif'))
    length = len(images)

    for i, im in enumerate(images):
        word = re.search('\/\d+-\d+-\d+-\d+-(.+?).tif', im).group(1)

        if d.check(word) and os.stat(im).st_size != 0:
            outpath = os.path.join(
                output, '%s_%s_%s.png' % (word, datasetNum, time.time()))
            Image.open(im).save(outpath)

        print('%s / %s' % (i, length), end='\r')
    print("Number of words:", len([n for n in os.listdir(output)]))

dataset_preprocessing(arg_folder1, arg_output)
dataset_preprocessing(arg_folder2, arg_output)

