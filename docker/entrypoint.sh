#!/bin/sh
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting API..."
exec uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port "${API_PORT}" --no-access-log
