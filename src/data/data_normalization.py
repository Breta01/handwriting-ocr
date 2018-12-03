import glob
import numpy as np
import cv2
from PIL import Image
import os
import sys

location = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(location, '../'))
from ocr.normalization import word_normalization
from ocr.viz import print_progress_bar
from .data_extractor import datasets


data_folder = 'words_final'
output_folder = os.path.join(location, '../../data/processed/')


parser = argparse.ArgumentParser(description='Script normalizing words from datasts.')
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


def words_norm(location, output):
    output = os.path.join(location, output)
    if not os.path.exists(output):
        os.makedirs(output)
    else:
        print("THIS DATASET IS BEING SKIPPED")
        print("Output folder already exists:", output)
        return 1        
        
    imgs = glob.glob(os.path.join(location, data_folder, '*.png'))
    length = len(imgs)

    for i, img_path in enumerate(imgs):
        image = cv2.imread(img_path)
        # Simple check for invalid images
        if image.shape[0] > 20:
            cv2.imwrite(
                os.path.join(output, os.path.basename(img_path)),
                word_normalization(
                    image,
                    height=64,
                    border=False,
                    tilt=False,
                    hystNorm=False))
        print_progress_bar(i, len(imgs))
        
    print("\tNumber of normalized words:",
          len([n for n in os.listdir(output)]))

    
if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == 'all':
        args.dataset = datasets.keys()[:-1]

    assert args.path == [] or len(args.dataset) == len(args.path), "provide same number of paths as datasets (use '' for default)"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for ds in args.dataset:
        print("Processing -", ds)
        entry = datasets[ds]
        words_norm(entry[1], os.path.join(output_folder, ds)
