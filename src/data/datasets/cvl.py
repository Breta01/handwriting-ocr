import enchant
import glob
import os
import sys
import time
import re
from PIL import Image
# Allow accesing files relative to this file
location = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(location, '../../'))
from ocr.viz import print_progress_bar


def extract(location, output, number=3):
    output = os.path.join(location, output)
    if not os.path.exists(output):
        os.makedirs(output)

    for sub in ['cvl-database-1-1/testset', 'cvl-database-1-1/trainset']:
        folder = os.path.join(location, sub)
        images = glob.glob(os.path.join(folder, 'words', '*', '*.tif'))

        d = enchant.Dict('en_US')

        for i, im in enumerate(images):
            word = re.search('\/\d+-\d+-\d+-\d+-(.+?).tif', im).group(1)

            if d.check(word) and os.stat(im).st_size != 0:
                outpath = os.path.join(
                    output,
                    '%s_%s_%s.png' % (word, number, time.time()))
                Image.open(im).save(outpath)
            print_progress_bar(i, len(images))

    print("\tNumber of words:", len([n for n in os.listdir(output)]))
