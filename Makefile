MODULE_FOLDER := GPTagger

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
