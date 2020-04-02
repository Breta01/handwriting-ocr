# Handwriting OCR
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The project tries to create software for recognition of a handwritten text from photos (also for Czech language). It uses computer vision and machine learning. And it experiments with different approaches to the problem. It started as a school project which I got a chance to present on Intel ISEF 2018.

<p align="center"><img src ="doc/imgs/poster.png?raw=true" height="340" alt="Sublime's custom image" /></p>

## Program Structure
Proces of recognition is divided into 4 steps. The initial input is a photo of page with text.

1. Detection of page and removal of background
2. Detection and separation of words
3. Normalization of words
4. Separation and recegnition of characters (recognition of words)

Main files combining all the steps are [OCR.ipynb](notebooks/OCR.ipynb) or [OCR-Evaluator.ipynb](notebooks/ocr_evaluator.ipynb). Naming of files goes by step representing - name of machine learning model.

## Getting Started
### 1. Clone the repository
```bash
git clone https://github.com/Breta01/handwriting-ocr.git
```
After downloading the repo, you have to download the datasets and models (for more info look into [data](data/) and [models](models/) folders).

### 2. Requirements
```bash
make bootstrap
```
The project is using Python 3.7 with Jupyter Notebook. I recommend using virtualenv. Running command `make bootstrap` should install all necessary packages. If you have it, you can run the installation as:

Main libraries (all required libraries are in [requirements.txt](requirements.txt) and [requirements-dev.txt](requirements-dev.txt)):
* Numpy
* Tensorflow
* OpenCV
* Pandas
* Matplotlib

### Activate and Run
```bash
source venv/bin/activate
```
This command will activate the virtualenv. Then you can run `jupyter notebook` in the directory of the project and work on the particular notebook.

## Contributing
Best way how to get involved is through creating [GitHub issues](https://github.com/Breta01/handwriting-ocr/issues) or solving one! If there aren't any issues you can contact me directly on email.

## License
[MIT](./LICENSE.md)

## Support the project
If this project helped you or you want to support quick answers to questions and issues. Or you just think it is an interesting project. Please consider a small donation.

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif)](https://paypal.me/bretahajek/5)
