# Copyright 2020 Břetislav Hájek <info@bretahajek.com>
# Licensed under the MIT License. See LICENSE for details.
"""Modelu for creating sets (train/dev/test) of normalized images."""
# TODO: Rename to data_create_sets.py or something like that
import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from handwriting_ocr.data.data_loader import DATASETS, DATA_FOLDER
from handwriting_ocr.ocr.normalization import word_normalization


def words_norm(location, output):
    output = os.path.join(location, output)
    if os.path.exists(output):
        print("THIS DATASET IS BEING SKIPPED")
        print("Output folder already exists:", output)
        return 1
    else:
        output = os.path.join(output, "words_nolines")
        os.makedirs(output)

    imgs = glob.glob(os.path.join(location, data_folder, "*.png"))
    length = len(imgs)

    for i, img_path in enumerate(imgs):
        image = cv2.imread(img_path)
        # Simple check for invalid images
        if image.shape[0] > 20:
            cv2.imwrite(
                os.path.join(output, os.path.basename(img_path)),
                word_normalization(
                    image, height=64, border=False, tilt=False, hyst_norm=False
                ),
            )
        print_progress_bar(i, len(imgs))

    print("\tNumber of normalized words:", len([n for n in os.listdir(output)]))


def create_sets(data_path, test_size, dev_size, seed=42):
    np.random.seed(seed)
    lines = []
    for d in DATASETS:
        if d.is_downloaded(data_path):
            lines.extend(d.load(data_path))
    np.random.shuffle(lines)

    test_i, dev_i = map(
        int, len(lines) * test_size, len(lines) * (test_size + dev_size)
    )

    sets = {
        "test": lines[:test_i],
        "dev": lines[test_i:dev_i],
        "train": lines[dev_i:],
    }

    for k, lns in sets.items():
        folder = data_path.joinpath("sets", k)
        folder.mkdir(parents=True, exists_ok=True)
        # TODO: Normalize image and move it to folder
        # TODO: Create txt file with paths (image names) and labels


def get_args():
    parser = argparse.ArgumentParser(
        description="Script for creating sets (train/dev/test) of normalized images."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Percentage (0-1) size of test set.",
    )
    parser.add_argument(
        "--dev_size",
        type=float,
        default=0.1,
        help="Percentage (0-1) size of development set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random shuffle.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=DATA_FOLDER,
        help="Path of data folder (default is recommended).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for d in DATASETS:
        d.download(DATA_FOLDER)

    create_sets(args.data_path, args.test_size, args.dev_size, args.seed)
