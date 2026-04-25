.PHONY: run stop logs clean dev

run:
	docker compose up postgres minio minio-init mlflow -d --build

stop:
	docker compose down

logs:
	docker compose logs -f postgres minio mlflow

clean:
	docker compose down --volumes --remove-orphans

dev:
	$(MAKE) run
	uv run alembic upgrade head
	uv run run_local.py
