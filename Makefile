.PHONY: run stop logs clean dev test test-unit test-e2e benchmark populate inference

run:
	docker compose up postgres redis minio minio-init mlflow api prometheus grafana -d --build

stop:
	docker compose down

logs:
	docker compose logs -f postgres redis minio mlflow

clean:
	docker compose down --volumes --remove-orphans

dev:
	docker compose up postgres redis minio minio-init mlflow -d --build
	uv run alembic upgrade head
	uv run run_local.py

test-unit:
	docker compose --profile test build test
	docker compose --profile test run --rm --no-deps test pytest tests/ --ignore=tests/e2e -v --cov=src --cov-report=term-missing

test-e2e:
	docker compose --profile test build test
	docker compose --profile test run --rm test

test: test-unit test-e2e

populate:
	docker compose --profile populate run --rm populate

inference:
	docker compose --profile inference run --rm inference

benchmark:
	docker compose --profile benchmark run --rm benchmark
