test:
	pytest tests/

integration_test: build
	bash tests/integration_test/run.sh

quality_checks:
	isort .
	black .
	pylint --recursive=y .
