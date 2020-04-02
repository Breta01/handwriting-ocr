# Copyright 2020 Břetislav Hájek <info@bretahajek.com>
# Licensed under the MIT License. See LICENSE for details.
from abc import ABCMeta, abstractmethod
import os
from pathlib import Path
import sys
import urllib.request

import gdown
from tqdm import tqdm


class Progressbar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, path, username=None, password=None):
    if "drive.google.com" in url:
        gdown.download(url, output, quiet=False)
        return

    if username or password:
        # create a password manager
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, url, username, password)
        handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        # create "opener"
        opener = urllib.request.build_opener(handler)
        urllib.request.install_opener(opener)

    with Progressbar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=path, reporthook=t.update_to)


class Data(metaclass=ABCMeta):
    self.username, self.password = None, None

    @abstractmethod
    def load(self, data_path):
        pass

    def download(self, data_path):
        print(f"Downloading dataset {self.name}...")
        for url, f, folder in self.files:
            folder = data_path.joinpath(folder)
            output = folder.joinpath(f)
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
                download_url(url, output, self.username, self.password)
                if output.suffix in [".zip", ".gz", ".tgz"]:
                    self.file_extract(output)

    def file_extract(self, file_path):
        print(f"Extracting {file_path} file...")
        out_path = file_path.with_suffix("")

        if file_path.suffix == ".zip":
            oepn_file = lambda x: zipfile.ZipFile(x, "r")
        elif file_path.suffix in [".gz", ".tgz"]:
            open_file = lambda x: tarfile.open(x, "r:gz")

        with open_file(self.name) as data_file:
            data_file.extractall(out_path, data_dir)


class Breta:
    def __init__(self, name="breta"):
        self.name = name
        self.files = [
            (
                "https://drive.google.com/uc?id=1p7tZWzK0yWZO35lipNZ_9wnfXRNIZOqj",
                "data.zip",
                "raw",
            )(
                "https://drive.google.com/uc?id=1y6Kkcfk4DkEacdy34HJtwjPVa1ZhyBgg",
                "data.zip",
                "processed",
            )
        ]

    def load(self, data_path):
        pass


if __name__ == "__main__":
    Breta().download()
