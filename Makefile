.PHONY: tests
.DEFAULT_GOAL := ci

pip:
	@pip install -r "requirements.txt"
	@pip install -U pytest pytest-cov hypothesis

tests:
	@python -m pytest --cov-config .coveragerc --cov=./ tests/ --disable-warnings

ci: pip tests
