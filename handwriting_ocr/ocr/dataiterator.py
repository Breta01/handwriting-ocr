# -*- coding: utf-8 -*-
"""Classes for feeding data during training."""
import numpy as np
import pandas as pd
from .helpers import img_extend
from .datahelpers import sequences_to_sparse


class BucketDataIterator:
    """Iterator for feeding CTC model during training."""

    def __init__(
        self,
        images,
        targets,
        num_buckets=5,
        slider=(60, 30),
        augmentation=None,
        dropout=0.0,
        train=True,
    ):

        self.train = train
        self.slider = slider
        self.augmentation = augmentation
        self.dropout = dropout
        for i in range(len(images)):
            images[i] = img_extend(
                images[i], (self.slider[0], max(images[i].shape[1], self.slider[1]))
            )
        in_length = [image.shape[1] for image in images]

        # Create pandas dataFrame and sort it by images width (length)
        self.dataFrame = (
            pd.DataFrame({"in_length": in_length, "images": images, "targets": targets})
            .sort_values("in_length")
            .reset_index(drop=True)
        )

        bsize = int(len(images) / num_buckets)
        self.num_buckets = num_buckets
        self.buckets = []
        for bucket in range(num_buckets - 1):
            self.buckets.append(
                self.dataFrame.iloc[bucket * bsize : (bucket + 1) * bsize]
            )
        self.buckets.append(self.dataFrame.iloc[(num_buckets - 1) * bsize :])

        self.buckets_size = [len(bucket) for bucket in self.buckets]
        self.cursor = np.array([0] * num_buckets)
        self.bucket_order = np.random.permutation(num_buckets)
        self.bucket_cursor = 0
        self.shuffle()
        print("Iterator created.")

    def shuffle(self, idx=None):
        """Shuffle idx bucket or each bucket separately."""
        for i in [idx] if idx is not None else range(self.num_buckets):
            self.buckets[i] = self.buckets[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0

    def next_batch(self, batch_size):
        """Creates next training batch of size.
        Args:
            batch_size: size of next batch
        Retruns:
            (images, labels, images lengths, labels lengths)
        """
        i_bucket = self.bucket_order[self.bucket_cursor]
        # Increment cursor and shuffle in case of new round
        self.bucket_cursor = (self.bucket_cursor + 1) % self.num_buckets
        if self.bucket_cursor == 0:
            self.bucket_order = np.random.permutation(self.num_buckets)

        if self.cursor[i_bucket] + batch_size > self.buckets_size[i_bucket]:
            self.shuffle(i_bucket)

        # Handle too big batch sizes
        if batch_size > self.buckets_size[i_bucket]:
            batch_size = self.buckets_size[i_bucket]

        res = self.buckets[i_bucket].iloc[
            self.cursor[i_bucket] : self.cursor[i_bucket] + batch_size
        ]
        self.cursor[i_bucket] += batch_size

        # PAD input sequence and output
        input_max = max(res["in_length"])

        input_imgs = np.zeros(
            (batch_size, self.slider[0], input_max, 1), dtype=np.uint8
        )
        for i, img in enumerate(res["images"]):
            input_imgs[i][:, : res["in_length"].values[i], 0] = img

        if self.train:
            input_imgs = self.augmentation.augment_images(input_imgs)
        input_imgs = input_imgs.astype(np.float32)

        targets = sequences_to_sparse(res["targets"].values)
        return input_imgs, targets, res["in_length"].values
