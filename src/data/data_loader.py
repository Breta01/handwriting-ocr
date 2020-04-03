# Copyright 2020 Břetislav Hájek <info@bretahajek.com>
# Licensed under the MIT License. See LICENSE for details.
from abc import ABCMeta, abstractmethod
import os
from pathlib import Path
import sys
import tarfile
import urllib.request
import zipfile

import gdown
from tqdm import tqdm


class Progressbar(tqdm):
    """Helper class for download progressbar."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output, username=None, password=None):
    if "drive.google.com" in url:
        gdown.download(url, str(output), quiet=False)
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
        urllib.request.urlretrieve(url, filename=output, reporthook=t.update_to)


class Data(metaclass=ABCMeta):
    """Abstract class for managing data."""

    username, password = None, None

    @abstractmethod
    def load(self, data_path):
        pass

    def download(self, data_path):
        print(f"Collecting dataset {self.name}...")
        for url, f, folder in self.files:
            folder = data_path.joinpath(folder, self.name)
            output = folder.joinpath(f)
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
                download_url(url, output, self.username, self.password)
                if output.suffix in [".zip", ".gz", ".tgz", ".tar"]:
                    self.file_extract(output)
                    output.unlink()

    def file_extract(self, file_path):
        print(f"Extracting {file_path} file...")
        if file_path.suffix == ".zip":
            open_file = lambda x: zipfile.ZipFile(x, "r")
        elif file_path.suffix in [".gz", ".tgz", ".tar"]:
            open_file = lambda x: tarfile.open(x, "r:gz")

        out_path = file_path.parent
        with open_file(file_path) as data_file:
            data_file.extractall(out_path)


class Breta(Data):
    """Handwriting data from Břetislav Hájek."""

    def __init__(self, name="breta"):
        self.name = name
        self.files = [
            (
                "https://drive.google.com/uc?id=1p7tZWzK0yWZO35lipNZ_9wnfXRNIZOqj",
                "data.zip",
                "raw",
            ),
            (
                "https://drive.google.com/uc?id=1y6Kkcfk4DkEacdy34HJtwjPVa1ZhyBgg",
                "data.zip",
                "processed",
            ),
        ]

    def load(self, data_path):
        pass


class CVL(Data):
    """CVL Database
    More info at: https://zenodo.org/record/1492267#.Xob4lPGxXeR
    """

    def __init__(self, name="cvl"):
        self.name = name
        self.files = [
            (
                "https://zenodo.org/record/1492267/files/cvl-database-1-1.zip",
                "cvl-database-1-1.zip",
                "raw",
            )
        ]

    def load(self, data_path):
        pass


class IAM(Data):
    """IAM Handwriting Database
    More info at: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self, name="iam"):
        # TODO: Require password (link reg. page), handle for errors
        self.name = name
        self.files = [
            (
                "http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/lines.txt",
                "lines.txt",
                "raw",
            ),
            (
                "http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz",
                "lines.tgz",
                "raw",
            ),
        ]

    def load(self, data_path):
        pass


class ORAND(Data):
    """ORAND CAR 2014 dataset
    More info at: https://www.orand.cl/icfhr2014-hdsr/#datasets
    """

    def __init__(self, name="orand"):
        self.name = name
        self.files = [
            (
                "https://www.orand.cl/orand_car/ORAND-CAR-2014.tar.gz",
                "ORAND-CAR-2014.tar.gz",
                "raw",
            )
        ]

    def load(self, data_path):
        pass


class Camb(Daa):
    """Cambridge Handwriting Database
    More info at: ftp://svr-ftp.eng.cam.ac.uk/pub/data/handwriting_databases.README
    """

    def __init__(self, name="camb"):
        self.name = name
        self.files = [
            (
                "ftp://svr-ftp.eng.cam.ac.uk/pub/data/handwriting_databases.README",
                "handwriting_databases.README",
                "raw",
            ),
            ("ftp://svr-ftp.eng.cam.ac.uk/pub/data/lob.tar", "lob.tar" "raw"),
            ("ftp://svr-ftp.eng.cam.ac.uk/pub/data/numbers.tar", "numbers.tar" "raw"),
        ]


if __name__ == "__main__":
    data_folder = Path(__file__).parent.joinpath("../../data/")

    datasets = [Breta(), CVL(), IAM(), ORAND(), Camb()]
    for d in datasets:
        d.download(data_folder)
