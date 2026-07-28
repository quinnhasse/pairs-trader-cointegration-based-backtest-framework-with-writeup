.PHONY: install backtest test lint clean

PYTHON := python3
OUTPUT_DIR := outputs

install:
	pip install -r requirements.txt

backtest: $(OUTPUT_DIR)
	$(PYTHON) scripts/run_backtest.py

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

test:
	pytest tests/ -v

lint:
	ruff check pairs_trader/ tests/

clean:
	rm -rf $(OUTPUT_DIR)/*.json $(OUTPUT_DIR)/*.png $(OUTPUT_DIR)/*.pdf
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
