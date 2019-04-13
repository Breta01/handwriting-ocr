FROM continuumio/anaconda3
ADD /Users/chris/handwriting-ocr/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml



