import argparse
import glob
import os
import random
import sys
from shutil import copyfile

import cv2
import numpy as np

location = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(location, '../'))

from create_csv import create_csv
from data_extractor import datasets
from ocr.viz import print_progress_bar


random.seed(17)  # Make the datasets split random, but reproducible
data_folder = 'words_final'
output_folder = os.path.join(location, '../../data/sets/')

# Sets percent distribution
test_set = 0.1
validation_set = 0.1


parser = argparse.ArgumentParser(
    description='Script sliting processed words into train, validation and test sets.')
parser.add_argument(
    '-d', '--dataset',
    nargs='*',
    choices=datasets.keys(),
    help='Pick dataset(s) to be used.')
parser.add_argument(
    '-p', '--path',
    nargs='*',
    default=[],
    help="""Path to folder containing the dataset. For multiple datasets
    provide path or ''. If not set, default paths will be used.""")
parser.add_argument(
    '--output',
    default='data-handwriting/sets',
    help="Directory for normalized and split data")
parser.add_argument(
    '--csv',
    action='store_true',
    default=False,
    help="Include flag if you want to create csv files along with split.")


if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == ['all']:
        args.dataset = list(datasets.keys())[:-1]

    assert args.path == [] or len(args.dataset) == len(args.path), \
        "provide same number of paths as datasets (use '' for default)"
    if args.path != []:
        for ds, path in zip(args.dataset, args.path):
            datasets[ds][1] = path

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    imgs = []
    for ds in args.dataset:
        for loc, _, _ in os.walk(datasets[ds][1].replace("raw", "processed")):
            imgs += glob.glob(os.path.join(loc, '*.png'))

    imgs.sort()
    random.shuffle(imgs)
    
    length = len(imgs)
    sp1 = int((1 - test_set - validation_set) * length)
    sp2 = int((1 - test_set) * length)
    img_paths = {'train': imgs[:sp1], 'dev': imgs[sp1:sp2], 'test': imgs[sp2:]}
    
    i = 0
    for split in ['train', 'dev', 'test']:
        split_output = os.path.join(output_folder, split)
        if not os.path.exists(split_output):
            os.mkdir(split_output)
        for im_path in img_paths[split]:
            copyfile(im_path, os.path.join(split_output, os.path.basename(im_path)))
            if '_gaplines' in im_path:
                im_path = im_path[:-3] + 'txt' 
                copyfile(
                    im_path, os.path.join(split_output, os.path.basename(im_path)))

            print_progress_bar(i, length)
            i += 1

        print(
            "\n\tNumber of %s words: %s" % (split, len(os.listdir(split_output))))

    if args.csv:
        create_csv(output_folder)
