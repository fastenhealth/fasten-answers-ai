
format:
	@echo "Running code formatting and checks..."
	ruff format . && ruff check . --fix

lint:
	@echo "Running linter..."
	ruff format . --check && ruff check .
