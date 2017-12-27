# Handwriting OCR
The project tries to create software for recognition of a handwritten text from photos (also for Czech language). It uses computer vision and machine learning. And it experiments with different approaches to the problem.

## Program Structure
Proces of recognition is divided into 5 steps, starting with photo of page with text.

1. Detection of page and removal of background
2. Detection and separation of words
3. Normalization of words
4. Separation and recegnition of characters (recognition of words)

Main files combining all the steps are [OCR.ipynb](OCR.ipynb) and [OCR-Evaluator.ipynb](OCR-Evaluator.ipynb). Naming of files goes by step representing - name of machine learning model. Notebooks ending with `DM` stands for creation of dataset.

## Getting Started
### Requirements
The project is created using Python 3.6 with Jupyter Notebook. Main libraries:
* Numpy (1.13)
* Tensorflow (1.4)
* OpenCV (3.1)
* Pandas (0.21)
* Matplotlib (2.1)

### Run
With all required libraries installed clone the repo and run `jupyter notebook` in the directory of the project. Then you can work on the particular notebook.

## Contributing
Best way how to get involved is through creating [GitHub issues](https://github.com/Breta01/handwriting-ocr/issues) or solving one! If there aren't any issues you can contact me directly on email.

## License
**MIT**
