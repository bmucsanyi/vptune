.DEFAULT: help

help:
	@echo "venv"
	@echo "        Create virtual environment"
	@echo "install"
	@echo "        Install vptune and dependencies"
	@echo "install-dev"
	@echo "        Install vptune and development tools"
	@echo "lint"
	@echo "        Run all linting actions"
	@echo "test"
	@echo "        Run pytest on test and report coverage"
	@echo "ruff-format"
	@echo "        Run ruff format on the project"
	@echo "ruff-format-check"
	@echo "        Check if ruff format would change files"
	@echo "ruff-check-fix"
	@echo "        Run ruff on the project and fix errors"
	@echo "ruff-check"
	@echo "        Run ruff check on the project without fixing errors"

.PHONY: venv

venv:
	@uv venv --python=3.13

.PHONY: install

install:
	@uv sync

.PHONY: install-dev

install-dev:
	@uv sync --extra dev
	@uv run pre-commit install

.PHONY: test

test:
	@uv run --no-sync pytest -qx --cov=src

.PHONY: ruff-format ruff-format-check

ruff-format:
	@uv run --no-sync ruff format src tests

ruff-format-check:
	@uv run --no-sync ruff format --check src tests

.PHONY: ruff-check

ruff-check-fix:
	@uv run --no-sync ruff check src tests --fix

ruff-check:
	@uv run --no-sync ruff check src tests

.PHONY: lint

lint:
	make ruff-format-check
	make ruff-check

.PHONY: lint-fix

lint-fix:
	@uv run --no-sync ruff format src tests
	@uv run --no-sync ruff check src tests --fix
	@uv run --no-sync ty check src tests
