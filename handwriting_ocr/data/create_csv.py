import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np
import simplejson

location = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(location, "../"))
from ocr.viz import print_progress_bar


parser = argparse.ArgumentParser()
parser.add_argument(
    "--sets",
    default=os.path.join(location, "../../data/sets/"),
    help="Folder with sets for converting to CSV.",
)


def create_csv(datadir):
    print("Converting word images to CSV...")
    img_paths = {
        "train": glob.glob(os.path.join(datadir, "train", "*.png")),
        "dev": glob.glob(os.path.join(datadir, "dev", "*.png")),
        "test": glob.glob(os.path.join(datadir, "test", "*.png")),
    }

    for split in ["train", "dev", "test"]:
        labels = np.array(
            [os.path.basename(name).split("_")[0] for name in img_paths[split]]
        )
        length = len(img_paths[split])
        images = np.empty(length, dtype=object)

        for i, img in enumerate(img_paths[split]):
            gaplines = "None"
            if os.path.isfile(img[:-3] + "txt"):
                with open(img[:-3] + "txt", "r") as fp:
                    gaplines = str(simplejson.load(fp))[1:-1]
            images[i] = (cv2.imread(img, 0), gaplines)
            print_progress_bar(i, length)

        with open(os.path.join(datadir, split + ".csv"), "w") as csvfile:
            fieldnames = ["label", "shape", "image", "gaplines"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(length):
                writer.writerow(
                    {
                        fieldnames[0]: labels[i],
                        fieldnames[1]: str(images[i][0].shape)[1:-1],
                        fieldnames[2]: str(list(images[i][0].flatten()))[1:-1],
                        fieldnames[3]: images[i][1],
                    }
                )

    print("\tCSV files created!")


if __name__ == "__main__":
    args = parser.parse_args()
    create_csv(args.sets)
