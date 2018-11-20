import time
import os
from shutil import copyfile

dirname = os.path.dirname(__file__)
arg_folder = os.path.join(dirname, '

arg_label_file = './../../data/raw/iam/words.txt'    # Location of words.txt file from IAM dataset
arg_folder = './../../data/raw/iam/words'            # Folder location of IAM dataset
arg_output = 'words-final'         # Output folder location
arg_erroutput = 'wordsIAM-err'  # Output with words with error separation

datasetNum = 2
# Words with these characters are removed
# you have to extend the alphabet in order to use them (ocr/datahelpers.py)
prohibited = [',', '(', ')', ';', ':', '/', '\\',
              '#', '"', '?', '!', '*', '_', '&']

def dataset_preprocessing(label_file, folder, output, err_output):
    length = len(open(label_file).readlines())

    if not os.path.exists(output):
        os.makedirs(output)
    if not os.path.exists(err_output):
        os.makedirs(err_output)

    with open(label_file) as fp:
        print("Processing files:")
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
                        out, "%s_%s_%s.png" % (word, datasetNum, time.time()))
                    copyfile(impath, outpath)

            print('%s / %s' % (i, length), end = '\r')
        print("Number of words:", len([n for n in os.listdir(output)]))

dataset_preprocessing(arg_label_file, arg_folder, arg_output, arg_erroutput)
