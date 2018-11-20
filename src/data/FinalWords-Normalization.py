import glob
import numpy as np
import cv2
from PIL import Image
import os
from ocr.normalization import imageNorm
from ocr.helpers import resize

arg_data_folder = 'data-handwriting'
arg_output = 'words-norm'

def words_norm(folder, output):
    if not os.path.exists(os.path.join(folder, output)):
        os.makedirs(os.path.join(folder, output))
    # imgs = glob.glob(os.path.join(folder, '*/words-final/*.png'))
    imgs = glob.glob(os.path.join(folder, 'IAM/words-final/*.png'))
    length = len(imgs)

    for i, im_path in enumerate(imgs):
        im = cv2.imread(im_path)
        if im.shape[0] > 20:
            cv2.imwrite(
                os.path.join(folder, output, os.path.basename(im_path)),
                imageNorm(im, height=60, border=False, tilt=False, hystNorm=False))

        if i % 100 == 0:
            print('%s / %s' % (i, length), end = '\r')
    print("Number of normalized words:",
          len([n for n in os.listdir(os.path.join(folder, output))]))

words_norm(arg_data_folder, arg_output)
