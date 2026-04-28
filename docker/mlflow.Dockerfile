FROM python:3.13-slim-bookworm

RUN pip install --no-cache-dir \
    mlflow==3.11.1 \
    boto3==1.42.96 \
    psycopg2-binary==2.9.12

EXPOSE 5000
