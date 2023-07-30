.DEFAULT_GOAL := help

sources = src/
tests = tests/

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: clean
clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

.PHONY: clean-build
clean-build: ## remove parser build
	rm -rf dist

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -not -path '*/.venv/*' -exec rm -f {} +
	find . -name '*.pyo' -not -path '*/.venv/*' -exec rm -f {} +
	find . -name '*~' -not -path '*/.venv/*' -exec rm -f {} +
	find . -name '__pycache__' -not -path '*/.venv/*' -exec rm -fr {} +

.PHONY: clean-test
clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -f .coverage.*
	rm -f coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr .ruff_cache
	rm -fr testtemp/
	rm -fr .tox/

.PHONY: lint-format
lint-format: ## lint and format code
	poetry run tox -e "pre-commit, mypy"

.PHONY: test
test: ## run pytest
	poetry run tox -e "py310, py311"

.PHONY: get-changelog
get-changelog: ## list all commits since last tag
	git log $$(git describe --tags --abbrev=0)..@ --pretty=%B --no-merges
