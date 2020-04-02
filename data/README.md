>**Disclaimer:** Unless otherwise stated I don't own any rights to data linked on this page. All usage of external data must comply with their licensing.
# Datasets
This folder should contain all data used in the project. Most of the data are too large to be included here, so you **have to download them and placed them according to instructions below.** If you have any suggestion or issues with the dataset, send me an email or create a new issue in the repository. 

## Words datasets
All together it should total in about 188000 images. After extraction, each word is named as: `<word>_<dataset-num>_<timestamp>.png`. The way of labelling can be changed, currently '_' is prohibited in all words. The number of a dataset is written behind each name in brackets. For example: `car_1_1528457794.9072268.png` - file corresponds to the image of a word 'car' from Breta’s data.

After downloading these datasets, there are scripts in `src/data/` folder which extract words from each dataset and place them in `words-final/` folder under each dataset folder. Another script takes all words from all `words-final/` folders and normalized them. Normalized words are placed in `processed/` folder. Finally, you run the script which separates data into training, validation and test sets, placing them into `sets/` folder.

### Breta’s data (1)
*5000 images*  
All data owned by [@Breta01](https://github.com/Breta01) are available on this link (distributed under the same license as this repository). The data should be placed either in `raw/breta/` or `processed/breta/` folder accordingly (see links below). (I removed the Czech accents from words. If you want to use them, you have to recover them using CSV files containing: `word_without_accents, original_word` in UTF-8 encoding.)

`raw/breta/`: <https://drive.google.com/file/d/1y6Kkcfk4DkEacdy34HJtwjPVa1ZhyBgg/view?usp=sharing>  
`processed/brata/`: <https://drive.google.com/file/d/1p7tZWzK0yWZO35lipNZ_9wnfXRNIZOqj/view?usp=sharing>

### IAM Handwriting Database (2)
*85000 images*  
The data can be obtained from the link below. First, you have to register (please read the terms and conditions). Then on the download page, from link `data/ascii` download `words.txt` file and from `data/words` download `words.tgz` file. Both of these files should be placed in `raw/iam/` folder. Then you should extract the `words.tgz` archive into `words/` folder.

<http://www.fki.inf.unibe.ch/databases/iam-handwriting-database>

### CVL (3)
*84000 images*  
First, read the terms and conditions. After that, you can download `cvl-database-1-1.zip` archive. Place this archive into `raw/cvl/` folder and extract it (it should create `cvl-database-1-1/` folder).

<https://zenodo.org/record/1492267>


### ORAND CAR 2014 (4)
*11700 images (only number strings)*  
Once again read the licensing. Then you can download `ORAND-CAR-2014.tar.gz` archive and place it into `raw/orand/` folder. Here you should extract the archive, resulting in `ORAND-CAR-2014/` folder.

<http://www.orand.cl/en/icfhr2014-hdsr/#datasets>


### Cambridge Handwriting Database (5)
*5200 images*  
First, read the README file (the first link). After that, you can download `lob.tar` and `numbers.tar` archives and place them into `raw/camb/` folder. Here you should extract each archive into the corresponding folder `lob.tar` to `lob/` and `numbers.tar` to `numbers/`.

<ftp://svr-ftp.eng.cam.ac.uk/pub/data/handwriting_databases.README>  
<ftp://svr-ftp.eng.cam.ac.uk/pub/data/>

## Dictionaries
`dictionaries/` folder contains a list of most common words for simple autocorrection.

## Pages
It contains photos of pages which are used for testing of page detection, words detection and testing in general.

## Characters Dataset
`characters/` folder contains single character images. There is only a few of them and they are already pre-processed. I advise using different dataset (for example NIST Special Database 19). Currently, there is no pre-processing script for it.
* NIST Special Database 19: <https://www.nist.gov/srd/nist-special-database-19>
* MNIST - the legend (only numbers): <http://yann.lecun.com/exdb/mnist/>

