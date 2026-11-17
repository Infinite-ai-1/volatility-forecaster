PYTHON ?= python
.PHONY: setup fetch backtest final-test reproduce test lint
setup:
	$(PYTHON) -m pip install -r requirements.txt
fetch:
	$(PYTHON) -m src.fetch_data
backtest:
	$(PYTHON) -m src.run_backtest
final-test:
	$(PYTHON) -m src.run_final_test
reproduce:
	$(PYTHON) -m src.reproduce
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
