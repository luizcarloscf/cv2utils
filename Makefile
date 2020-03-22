PYTHON := python3
PIP := $(PYTHON) -m pip

.PHONY: clean-pyc clean-build docs clean test

help:
	@ echo "Usage:\n"
	@ echo "make clean                remove all build, test, coverage and Python artifacts"
	@ echo "make clean-build          remove build artifacts"
	@ echo "make clean-pyc            remove Python file artifacts"
	@ echo "make clean-test           remove test and coverage artifacts"
	@ echo "make test                 use Tox to run test and flake8"
	@ echo "make docs                 generate Sphinx HTML documentation, including API docs"
	

clean: clean-build clean-pyc clean-test

clean-build:
	@ rm -fr build/
	@ rm -fr dist/
	@ rm -fr .eggs/
	@ find . -name '*.egg-info' -exec rm -fr {} +
	@ find . -name '*.egg' -exec rm -rf {} +

clean-pyc:
	@ find . -name '*.pyc' -exec rm -f {} +
	@ find . -name '*.pyo' -exec rm -f {} +
	@ find . -name '*~' -exec rm -f {} +
	@ find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	@ rm -fr .tox/
	@ rm -f .coverage

test:
	@ $(PYTHON) -m tox

docs:
	$(MAKE) -C docs clean
	$(MAKE) -C docs html 
