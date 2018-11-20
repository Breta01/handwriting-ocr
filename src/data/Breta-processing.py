import os
from PIL import Image
from shutil import copyfile
import enchant
import time

arg_folder = './../../data/raw/breta_words'
arg_output = 'words-final'

datasetNum = 1

def dataset_preprocessing(folder, output):
    length = len([n for n in os.listdir(folder)])
    d = enchant.Dict('en_US')
    if not os.path.exists(output):
        os.makedirs(output)

    for i, data in enumerate(os.listdir(folder)):
        word = data.split('_')[0]
        if d.check(word):
            im = os.path.join(folder, data)
            out = os.path.join(output, '%s_%s_%s.png' % (word, datasetNum, time.time()))
            Image.open(im).save(out)

        print('%s / %s' % (i, length), end='\r')
    print("Number of words:", len([n for n in os.listdir(output)]))

dataset_preprocessing(arg_folder, arg_output)
