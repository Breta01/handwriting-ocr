# Copyright 2020 Břetislav Hájek <info@bretahajek.com>
# Licensed under the MIT License. See LICENSE for details.
"""Modelu for downloading datasets and loading data (DATASETS list)."""

import xml.etree.ElementTree
from abc import ABCMeta, abstractmethod
from pathlib import Path

import cv2 as cv
import numpy as np
from handwriting_ocr.data.loader import Loader

DATA_FOLDER = Path(__file__).parent.joinpath("../../data/")


class Dataset(Loader):
    """Abstract class for managing data.

    Attributes:
        name (str): Name of dataset. It is used for naming appropriate folders.
        files (List[Tuple[str, str, str, str]]): List of datasets' files/folders (URL,
            tmp file, final file or folder, dataset type folder)
        require_auth (bool): If authentication is required (default = False)
        username (str): (Optional) username for authentication during donwload
        password (str): (Optional) password for authentication during donwload
    """

    @abstractmethod
    def load(self, data_path):
        """Returns path to line images with corresponding labels (sorted by path).

        Args:
            data_path (Path): Path to data folder

        Returns:
            lines (List[Tuple[Path, str]]): List of tuples containing path to image and
                label. It should always return images in same order (sort on return).
        """
        ...

    def __str__(self):
        return f"dataset-{self.name}"


class Breta(Dataset):
    """Handwriting data from Břetislav Hájek."""

    name = "breta"
    # TODO: Remove data.zip archive (no longer in use)
    files = [
        (
            "https://drive.google.com/uc?id=1y6Kkcfk4DkEacdy34HJtwjPVa1ZhyBgg",
            "data.zip",
            "",
            "raw",
        ),
        (
            "https://drive.google.com/uc?id=1p7tZWzK0yWZO35lipNZ_9wnfXRNIZOqj",
            "data.zip",
            "",
            "processed",
        ),
    ]

    def __init__(self, name="breta"):
        self.name = name

    def load(self, data_path):
        folder = data_path / self.files[0][3] / self.name / self.files[0][2]
        return sorted((p, p.name.split("_")[0]) for p in folder.glob("**/*.png"))


class CVL(Dataset):
    """CVL Database
    More info at: https://zenodo.org/record/1492267#.Xob4lPGxXeR
    """

    name = "cvl"
    files = [
        (
            "https://zenodo.org/record/1492267/files/cvl-database-1-1.zip",
            "cvl-database-1-1.zip",
            "",
            "raw",
        )
    ]

    def __init__(self, name="cvl"):
        self.name = name

    def load(self, data_path):
        lines = []

        folder = data_path / self.files[0][3] / self.name / self.files[0][2]
        l_dic = {}
        for xf in folder.glob("**/xml/*.xml"):
            try:
                with open(xf, "r") as f:
                    root = xml.etree.ElementTree.fromstring(f.read())
            except:
                with open(xf, "r", encoding="iso-8859-15") as f:
                    root = xml.etree.ElementTree.fromstring(f.read())
            # Get tag schema
            tg = root.tag[: -len(root.tag.split("}", 1)[1])]
            for attr in root.findall(
                f".//{tg}AttrRegion[@attrType='2'][@fontType='2']"
            ):
                target = " ".join(
                    x.get("text") for x in attr.findall(f"{tg}AttrRegion[@text]")
                )
                if len(target) != 0:
                    l_dic[attr.get("id")] = target

        ln_f = folder / self.files[0][2]
        return sorted(
            (p, l_dic[p.with_suffix("").name])
            for p in ln_f.glob("**/lines/*/*.tif")
            if p.with_suffix("").name in l_dic
        )


class IAM(Dataset):
    """IAM Handwriting Database
    More info at: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    name = "iam"
    require_auth = True
    files = [
        (
            "http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/lines.txt",
            "lines.txt",
            "lines.txt",
            "raw",
        ),
        (
            "http://www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz",
            "lines.tgz",
            "lines",
            "raw",
        ),
    ]

    def __init__(self, name="iam", username=None, password=None):
        self.name = name
        self.username = username
        self.password = password

    def load(self, data_path):
        lines = []

        folder = data_path / self.files[0][3] / self.name
        with open(folder.joinpath(self.files[0][2]), "r") as f:
            labels = [l.strip() for l in f if l.strip()[0] != "#"]
            labels = map(lambda x: (x.split(" ")[0], x.split(" ", 8)[-1]), labels)
            l_dic = {im: label.replace("|", " ") for im, label in labels}

        ln_f = folder / self.files[1][2]
        return sorted((p, l_dic[p.with_suffix("").name]) for p in ln_f.glob("**/*.png"))


class ORAND(Dataset):
    """ORAND CAR 2014 dataset
    More info at: https://www.orand.cl/icfhr2014-hdsr/#datasets
    """

    name = "orand"
    files = [
        (
            "https://www.orand.cl/orand_car/ORAND-CAR-2014.tar.gz",
            "ORAND-CAR-2014.tar.gz",
            "",
            "raw",
        )
    ]

    def __init__(self, name="orand"):
        self.name = name

    def load(self, data_path):
        lines = []

        folder = data_path / self.files[0][3] / self.name / self.files[0][2]
        for label_f in folder.glob("**/CAR-*/*.txt"):
            im_folder = Path(str(label_f)[:-6] + "images")
            with open(label_f, "r") as f:
                labels = map(lambda x: x.strip().split("\t"), f)
                lines.extend((im_folder.joinpath(im), w) for im, w in labels)
        return sorted(lines)


class Camb(Dataset):
    """Cambridge Handwriting Database
    More info at: ftp://svr-ftp.eng.cam.ac.uk/pub/data/handwriting_databases.README
    """

    name = "camb"
    files = [
        (
            "ftp://svr-ftp.eng.cam.ac.uk/pub/data/handwriting_databases.README",
            "handwriting_databases.README",
            "handwriting_databases.README",
            "raw",
        ),
        ("ftp://svr-ftp.eng.cam.ac.uk/pub/data/lob.tar", "lob.tar", "lob", "raw"),
        (
            "ftp://svr-ftp.eng.cam.ac.uk/pub/data/numbers.tar",
            "numbers.tar",
            "numbers",
            "raw",
        ),
    ]

    def __init__(self, name="camb"):
        self.name = name

    def post_download(self, data_path):
        print(f"Running post-download processing on {self.name}...")
        folder = data_path / self.files[0][3] / self.name
        output = folder / "extracted"
        output.mkdir(parents=True, exist_ok=True)

        for i, seg_f in enumerate(sorted(folder.glob("**/*.seg"))):
            with gzip.open(seg_f.with_suffix(".tiff.gz"), "rb") as f:
                buff = np.frombuffer(f.read(), dtype=np.int8)
                image = cv.imdecode(buff, cv.IMREAD_UNCHANGED)

            with open(seg_f, "r") as f:
                f.readline()
                for line in f:
                    rect = list(map(int, line.strip().split(" ")[1:]))
                    word = line.split(" ")[0]
                    im = image[rect[2] : rect[3], rect[0] : rect[1]]

                    if 0 in im.shape:
                        continue
                    cv.imwrite(str(output.joinpath(f"{word}_{i:04}.png")), im)

    def load(self, data_path):
        folder = data_path / self.files[0][3] / self.name / "extracted"
        return sorted((p, p.name.split("_")[0]) for p in folder.glob("**/*.png"))


class NIST(Dataset):
    """NIST SD 19 - character dataset
    More info at: https://www.nist.gov/srd/nist-special-database-19
    """

    name = "nist"
    files = [
        (
            "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",
            "by_class.zip",
            "",
            "raw",
        ),
    ]

    def __init__(self, name="nist"):
        self.name = name

    def load(self, data_path):
        # TODO: Generate lines from NIST characters
        return []

    def load_characters(self, data_path):
        folder = data_path / self.files[0][3] / self.name / self.files[0][2]
        return sorted(
            (p, chr(int(p.name.split("_")[1], 16)))
            for p in folder.glob("**/trian_*/*.png")
        )


DATASETS = [Breta(), CVL(), IAM(), ORAND(), Camb(), NIST()]


if __name__ == "__main__":
    for d in DATASETS:
        d.download(DATA_FOLDER)
        print(d, len(d.load(DATA_FOLDER)))
