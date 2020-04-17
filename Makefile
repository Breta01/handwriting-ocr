.PHONY: help bootstrap data lint clean

SHELL=/bin/bash

VENV_NAME?=venv
VENV_BIN=$(shell pwd)/${VENV_NAME}/bin
VENV_ACTIVATE=source $(VENV_NAME)/bin/activate

PROJECT_DIR=handwriting_ocr

PYTHON=${VENV_NAME}/bin/python3

.DEFAULT: help
help:
	@echo "Make file commands:"
	@echo "    make bootstrap"
	@echo "        Prepare complete development environment"
	@echo "    make data"
	@echo "        Download and prepare data for training"
	@echo "    make lint"
	@echo "        Run pylint and mypy"
	@echo "    make clean"
	@echo "        Clean repository"

bootstrap:
	sudo xargs apt-get -y install < requirements-apt.txt
	python3.7 -m pip install pip
	python3.7 -m pip install virtualenv
	make venv
	${VENV_ACTIVATE}; pre-commit install

# Runs when the file changes
venv: $(VENV_NAME)/bin/activate
$(VENV_NAME)/bin/activate: setup.py requirements.txt requirements-dev.txt
	test -d $(VENV_NAME) || virtualenv -p python3.7 $(VENV_NAME)
	${PYTHON} -m pip install -U pip
	${PYTHON} -m pip install -e .[dev]
	touch $(VENV_NAME)/bin/activate

data:
	${PYTHON} ${PROJECT_DIR}/data/data_create_sets.py

lint: venv
# pylint supports pyproject.toml from 2.5 version. Switch to following cmd once updated:
# ${PYTHON} -m pylint src
	${PYTHON} -m pylint --extension-pkg-whitelist=cv2 --variable-rgx='[a-z_][a-z0-9_]{0,30}$' --max-line-length=88 src
	${PYTHON} -m flake8 src

clean:
	find . -name '*.pyc' -exec rm --force {} +
	rm -rf $(VENV_NAME) *.eggs *.egg-info dist build docs/_build .cache
