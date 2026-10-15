.PHONY: install install-paper test test-all lint format demo reproduce clean
DATA ?= data/S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv

install:            ## editable install with dev tools (flexible versions)
	pip install -e ".[dev]"

install-paper:      ## exact environment of the paper (needed to match the published numbers)
	pip install -r requirements-paper.txt && pip install -e . --no-deps

test:               ## fast tests
	pytest -m "not slow and not paper"

test-all:           ## all tests (slow model fits); add TG_STACK_DATA=<csv> for the paper checks
	pytest

lint:
	ruff check src tests scripts && ruff format --check src tests scripts

format:
	ruff check --fix src tests scripts && ruff format src tests scripts

demo:               ## smoke run on SYNTHETIC data (minutes, not hours)
	tg-stack --data data/synthetic_demo.csv --out results/demo --quick --no-strict --threads 2 \
	         --stages audit sem ablation

reproduce:          ## full reproduction of the paper (about 2-2.5 h on 2 CPU cores)
	tg-stack --data $(DATA) --out results

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info results/demo
