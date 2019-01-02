import os
from PIL import Image
import time
import sys
# Allow accesing files relative to this file
location = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(location, '../../'))
from ocr.viz import print_progress_bar


def extract(location, output, number=1):
    output = os.path.join(location, output)
    if not os.path.exists(output):
        os.makedirs(output)

    for sub in ['words', 'archive', 'cz_raw', 'en_raw']:
        folder = os.path.join(location, sub)

        img_list = os.listdir(os.path.join(folder))
        for i, data in enumerate(img_list):
            word = data.split('_')[0]
            img = os.path.join(folder, data)
            out = os.path.join(
                output,
                '%s_%s_%s.png' % (word, number, data.split('_')[-1][:-4]))
            Image.open(img).save(out)
            print_progress_bar(i, len(img_list))

    print("\tNumber of words:", len([n for n in os.listdir(output)]))
