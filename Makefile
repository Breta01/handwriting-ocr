.PHONY: help bootstrap lint clean

SHELL=/bin/bash

VENV_NAME?=venv
VENV_BIN=$(shell pwd)/${VENV_NAME}/bin
VENV_ACTIVATE=source $(VENV_NAME)/bin/activate

PYTHON=${VENV_NAME}/bin/python3

.DEFAULT: help
help:
	@echo "Make file commands:"
	@echo "    make bootstrap"
	@echo "        Prepare complete development environment"
	@echo "    make lint"
	@echo "        run pylint and mypy"
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
$(VENV_NAME)/bin/activate: requirements.txt requirements-dev.txt # Replace with setup.py
	test -d $(VENV_NAME) || virtualenv -p python3.7 $(VENV_NAME)
	${PYTHON} -m pip install -U pip
	${PYTHON} -m pip install -r requirements-dev.txt
	${PYTHON} -m pip install -r requirements.txt
#	Replace with setup.py
#	${PYTHON} -m pip install -e .
	touch $(VENV_NAME)/bin/activate

lint: venv
# pylint supports pyproject.toml from 2.5 version. Switch to following cmd once updated:
# ${PYTHON} -m pylint src
	${PYTHON} -m pylint --extension-pkg-whitelist=cv2 --variable-rgx='[a-z_][a-z0-9_]{0,30}$' --max-line-length=88 src
	${PYTHON} -m flake8 src

data:
	${PYTHON} src/data/data_loader.py

clean:
	find . -name '*.pyc' -exec rm --force {} +
	rm -rf $(VENV_NAME) *.eggs *.egg-info dist build docs/_build .cache
