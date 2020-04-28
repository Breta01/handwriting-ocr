# Copyright 2020 Břetislav Hájek <info@bretahajek.com>
# Licensed under the MIT License. See LICENSE for details.
"""Module for downloading data."""

import getpass
import gzip
import tarfile
import urllib.request
import zipfile
from abc import ABCMeta, abstractmethod

import gdown
from tqdm import tqdm


class Progressbar(tqdm):
    """Helper class for download progressbar."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output, username=None, password=None):
    """Download file from URL to output location.

    Args:
        url (str): URL for downloading the file
        output (Path): Path where should be downloaded file stored
        username (str): (Optional) username for authentication
        password (str): (Optional) username for authentication

    Returns:
        status (int): Returns status of download (200: OK, 401: Unauthorized,
            -1: Unknown)
    """

    if "drive.google.com" in url:
        gdown.download(url, str(output), quiet=False)
        return 200

    for _ in range(3):
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
            try:
                urllib.request.urlretrieve(url, filename=output, reporthook=t.update_to)
                return 200
            except urllib.error.HTTPError as e:
                if hasattr(e, "code") and e.code == 401:
                    return 401
                print(f"\nError occured during download:\n{e}")
                return -1


def file_extract(file_path, out_path):
    """Extract archive file into given location.

    Args:
        file_path (Path): Path of archive file
        out_path (Path): Path of folder where should be the file extracted
    """
    print(f"Extracting {file_path} file...")
    if file_path.suffix == ".zip":

        def open_file(x):
            return zipfile.ZipFile(x, "r")

    elif file_path.suffix in [".gz", ".tgz"]:

        def open_file(x):
            return tarfile.open(x, "r:gz")

    elif file_path.suffix == ".tar":

        def open_file(x):
            return tarfile.open(x, "r")

    with open_file(file_path) as data_file:
        data_file.extractall(out_path)


class Loader(metaclass=ABCMeta):
    """Abstract class for downloading data.

    Attributes:
        name (str): Name of data. It is used for naming appropriate folders.
        files (List[Tuple[str, str, str, str]]): List of files/folders (URL, tmp file,
            final file or folder, dataset type folder)
        require_auth (bool): If authentication is required (default = False)
        username (str): (Optional) username for authentication during donwload
        password (str): (Optional) password for authentication during donwload
    """

    require_auth = False
    username, password = None, None

    @property
    @abstractmethod
    def name(self):
        ...

    @property
    @abstractmethod
    def files(self):
        ...

    def clear(self, data_path):
        """Clear all downloaded files.

        Args:
            data_path (Path): Path to data folder
        """
        for _, _, res, folder in self.files:
            d = data_path / folder / self.name
            if not d.exists():
                shutil.rmtree(d)

    def is_downloaded(self, data_path):
        """Check if files are downloaded.

        Args:
            data_path (Path): Path to data folder

        Returns:
            is_downloaded (bool): True if files are downloaded (no folder missing)
        """
        for _, _, res, folder in self.files:
            if not data_path.joinpath(folder, self.name, res).exists():
                return False
        return True

    def download(self, data_path):
        print(f"Collecting {self}...")
        downloaded = False
        for url, f, res, folder in self.files:
            folder = data_path / folder / self.name
            tmp_output = folder / f
            res_output = folder / res

            if not res_output.exists():
                tmp_output.parent.mkdir(parents=True, exist_ok=True)
                # Try the authentication 3 times
                for i in range(3):
                    if self.require_auth:
                        if not self.username:
                            self.username = input(f"Username for {self}: ")
                        if not self.password:
                            self.password = getpass.getpass(f"Password for {self}: ")
                    status = download_url(url, tmp_output, self.username, self.password)
                    if status == 200:
                        break

                    if status == 401 and i < 2:
                        print("Invalid username or password, please try again.")
                    else:
                        print(f"{self} skipped.")
                        return
                downloaded = True

                if tmp_output.suffix in [".zip", ".gz", ".tgz", ".tar"]:
                    file_extract(tmp_output, res_output)
                    tmp_output.unlink()

        if downloaded:
            self.post_download(data_path)

    def post_download(self, data_path):
        """Run post-processing on downloaded data (e.g. cut lines from form images)

        Args:
            data_path (Path): Path to data folder
        """

    def __str__(self):
        return self.name
