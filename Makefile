.PHONY: install lint test reproduce train-registry docker-build

install:
	python -m pip install -r requirements-dev.txt

lint:
	ruff check src tests

test:
	pytest -q

reproduce:
	REGISTER_MODEL=false MLFLOW_TRACKING_URI=file:./mlruns dvc repro

train-registry:
	MLFLOW_TRACKING_URI=$${MLFLOW_TRACKING_URI:-http://localhost:5000} REGISTER_MODEL=true python -m src.train

docker-build:
	docker build -t iris-training:local .

