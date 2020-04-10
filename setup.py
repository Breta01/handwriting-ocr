# Copyright 2020 Břetislav Hájek <info@bretahajek.com>
# Licensed under the MIT License. See LICENSE for details.
import setuptools
from pathlib import Path


CURRENT_DIR = Path(__file__).parent


def get_long_description() -> str:
    return (
        (CURRENT_DIR / "README.md").read_text(encoding="utf8")
        + "\n\n"
        + (CURRENT_DIR / "CHANGELOG.md").read_text(encoding="utf8")
    )


setuptools.setup(
    name="handwriting-ocr",
    version="0.0.0",
    author="Břetislav Hájek",
    author_email="info@bretahajek.com",
    description="OCR tool for handwriting.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/Breta01/handwriting-ocr",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    keywords="handwriting ocr",
    license="MIT",
    install_requires=[
        # TODO: Add requirements.txt
    ],
    extras_require={
        "dev": [
            # TODO: Add requirements-dev.txt
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
