# Makefile
.PHONY: all-format setup-project

all-format:
	@echo "Formatting all Python files in directoy: $(DIR)"
	FILES=$$(find $(DIR) -name "*.py" -not -path "*/venv/*" | grep -v "__init__.py"); \
	uv run black $$FILES; \
	uv run isort $$FILES; \
	uv run autoflake --in-place --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports $$FILES; \
	echo "Formatting completed."

all-format-gcp:
	@echo "Formatting all Python files in directoy: $(DIR)"
	FILES=$$(find $(DIR) -name "*.py" -not -path "*/venv/*" | grep -v "__init__.py"); \
	black $$FILES --target-version py312; \
	isort $$FILES; \
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports $$FILES; \
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


setup-project-gcp:
	@echo "Setting up project environment..."
	sudo apt install python3.12-venv
	python3 -m venv .venv
	. .venv/bin/activate && \
		pip install -r requirements.txt  && \
		cd data/enwik8 && \
		wget --continue http://mattmahoney.net/dc/enwik8.zip && \
		python3 prep_enwik8.py && \
		cd ../.. && \
		python3 prepare_enwik8.py
	@echo "Project setup completed."

setup-project-nb:
	@echo "Setting up project environment..."
	cd data/enwik8 && \
		wget --continue http://mattmahoney.net/dc/enwik8.zip && \
		python prep_enwik8.py
	python prepare_enwik8.py
	@echo "Project setup completed."