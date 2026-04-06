.PHONY: install dev lint
install:
	pip install -r requirements.txt
dev:
	python main.py
lint:
	python -c "import ast,pathlib;[ast.parse(p.read_text()) for p in pathlib.Path('.').rglob('*.py')]"
