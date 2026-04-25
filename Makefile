.PHONY: run stop logs

run:
	docker compose up postgres minio minio-init mlflow -d --build

stop:
	docker compose down

logs:
	docker compose logs -f postgres minio mlflow
