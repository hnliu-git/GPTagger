MODULE_FOLDER := GPTagger
MODULE_VERSION := $(shell git tag | sort -V | tail -1)

init: pyproject.toml
	pip install --upgrade pip
	command -v poetry || pip install poetry
	poetry env use python

install: init
		poetry install

test: install
		poetry run pytest tests --cov
		poetry run coverage-badge -f -o res/coverage.svg

analysis: install
		poetry run flake8 $(MODULE_FOLDER)
		poetry run flake8 tests

tidy: install
		poetry run black --preview $(MODULE_FOLDER)
		poetry run black --preview tests

build: init
		sed -i '' "s/version = \"0.0.0\"/version = \"$(MODULE_VERSION)\"/g" pyproject.toml
		poetry version $(MODULE_VERSION)
		poetry build

publish: build
		poetry publish
