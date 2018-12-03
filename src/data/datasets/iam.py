import time
import os
import sys
from shutil import copyfile
# Allow accesing files relative to this file
location = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(location, '../../'))
from ocr.viz import print_progress_bar


# Words with these characters are removed
# you have to extend the alphabet in order to use them (ocr/datahelpers.py)
prohibited = [',', '(', ')', ';', ':', '/', '\\',
              '#', '"', '?', '!', '*', '_', '&']


def extract(location, output, number=2):
    output = os.path.join(location, output)
    err_output = os.path.join(location, 'words_with_error')
    if not os.path.exists(output):
        os.makedirs(output)
    if not os.path.exists(err_output):
        os.makedirs(err_output)

    folder = os.path.join(location, 'words')
    label_file = os.path.join(location, 'words.txt')
    length = len(open(label_file).readlines())

    with open(label_file) as fp:
        for i, line in enumerate(fp):
            if line[0] != '#':
                l = line.strip().split(" ")
                impath = os.path.join(
                    folder,
                    l[0].split('-')[0],
                    l[0].split('-')[0] + '-' + l[0].split('-')[1],
                    l[0] + '.png')
                word = l[-1]

                if (os.stat(impath).st_size != 0
                    and word not in ['.', '-', "'"]
                    and not any(i in word for i in prohibited)):

                    out = output if l[1] == 'ok' else err_output
                    outpath = os.path.join(
                        out, "%s_%s_%s.png" % (word, number, time.time()))
                    copyfile(impath, outpath)

            print_progress_bar(i, length)
    print("\tNumber of words:", len([n for n in os.listdir(output)]))
