# Makefile
.PHONY: all-format setup-project

all-format:
	@echo "Formatting all Python files in directoy: $(DIR)"
	FILES=$$(find $(DIR) -name "*.py" -not -path "*/venv/*" | grep -v "__init__.py"); \
	uv run black $$FILES; \
	uv run isort $$FILES; \
	uv run autoflake --in-place --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports $$FILES; \
	echo "Formatting completed."

setup-project:
	@echo "Setting up project environment..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv python pin 3.13
	uv sync
	cd data/enwik8 && \
		wget --continue http://mattmahoney.net/dc/enwik8.zip && \
		uv run prep_enwik8.py
	uv run prepare_enwik8.py
	@echo "Project setup completed."