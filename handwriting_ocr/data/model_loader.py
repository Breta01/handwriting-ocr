# Copyright 2020 Břetislav Hájek <info@bretahajek.com>
# Licensed under the MIT License. See LICENSE for details.
"""Modelu for downloading and providing pre-trained models."""

from handwriting_ocr.data.loader import Loader

MODEL_FOLDER = Path(__file__).parent.joinpath("../../models/")


class Models(Loader):
    """Download pre-trained models."""

    name = "models"
    files = [
        (
            "https://drive.google.com/open?id=1YbmsiJK3Wclfm6K8PrJuz-QROEKX1qis"
            "ocr-handwriting-models.zip",
            "",
            "",
        ),
    ]

    def __init__(self, name="models"):
        self.name = name
