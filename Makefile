
install:
	uv sync

run:
	uv run python -m src 

debug:
	uv run python -m pdb src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

lint:
	flake8 src
	mypy src    --warn-return-any \
	            --warn-unused-ignores \
	            --ignore-missing-imports \
	            --disallow-untyped-defs \
	            --check-untyped-defs

.PHONY: install run debug clean lint