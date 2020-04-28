# Copyright 2020 Břetislav Hájek <info@bretahajek.com>
# Licensed under the MIT License. See LICENSE for details.
"""Modelu for creating sets (train/dev/test) of normalized images."""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

from handwriting_ocr.data.data_loader import DATASETS, DATA_FOLDER
from handwriting_ocr.ocr.normalization import word_normalization


def create_sets(data_path, test_size, dev_size, seed=42):
    """Loads all data and process them into train/dev/test sets.

    It loads all available datasets from data_path, splits them into train/dev/test sets
    and normalize images. Labels are saved into labels.txt file as
    `{filename.png}\t{label}` (separated by tab)

    Args:
        data_path (Path): Data folder path
        test_size (float): Percentage of test images from all images (between 0-1)
        dev_size (float): Percentage of dev images from all images (between 0-1)
        seed (int): Seed value for reproducibility of split
    """
    np.random.seed(seed)
    lines = []
    for d in DATASETS:
        if d.is_downloaded(data_path):
            lines.extend(d.load(data_path))
    np.random.shuffle(lines)

    test_i, dev_i = (len(lines) * np.array([test_size, test_size + dev_size])).astype(
        int
    )

    sets = {
        "test": lines[:test_i],
        "dev": lines[test_i:dev_i],
        "train": lines[dev_i:],
    }

    for k, set_lines in sets.items():
        print(f"{k} images: {len(set_lines)}")

        folder = data_path / "sets" / k
        folder.mkdir(parents=True, exist_ok=True)

        label_file = folder.joinpath("labels.txt").open("w+")
        for i, (path, label) in enumerate(tqdm(set_lines)):
            image = cv.imread(str(path))
            if image.shape[0] < 15:
                continue

            norm = word_normalization(
                image, height=64, border=False, tilt=False, hyst_norm=False
            )

            out_name = f"image_{i:05}.png"
            cv.imwrite(str(folder.joinpath(out_name)), norm)
            label_file.write(f"{out_name}\t{label}\n")

        label_file.close()


def get_args():
    """ArgumentParser for sets creation."""
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
