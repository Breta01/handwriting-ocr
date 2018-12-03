import argparse
import glob
import os
import random
import numpy as np
import cv2

location = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(location, '../'))
from ocr.viz import print_progress_bar
from .data_extractor import datasets
from .create_csv import create_csv


random.seed(17)  # Make the datasets split random and reproducible
data_folder = 'words_final'
output_folder = os.path.join(location, '../../data/sets/')


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
    folder = args.data

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # imgs = glob.glob(os.path.join(folder, '*/words-final/*.png'))
    imgs = []
    for ds in args.dataset:
        for loc, _, _ in os.walk(os.path.join(folder, ds)):
            imgs += glob.glob(os.path.join(loc, '*.png'))

    imgs.sort()
    random.shuffle(imgs)
    
    length = len(imgs)
    sp1 = int(0.8 * length)
    sp2 = int(0.9 * length)
    img_paths = {'train': imgs[:sp1], 'dev': imgs[sp1:sp2], 'test': imgs[sp2:]}
    
    i = 0
    for split in ['train', 'dev', 'test']:
        split_output = os.path.join(output_folder, split)
        if not os.path.exists(split_output):
            os.mkdir(split_output)
        for im_path in img_paths[split]:
            # Copy image
            print_progress_bar(i, length)
            i += 1
        print(
            "\tNumber of %s words: %s" % (split, len(os.listdir(split_output))))

    if args.csv:
        create_csv(output_folder)
