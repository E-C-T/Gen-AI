FROM python:3.9-slim

RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./package[s] ./packages

COPY ./.env ./

COPY ./GCP-generativeai-Key.json ./

RUN poetry install  --no-interaction --no-ansi --no-root

RUN apt-get update && apt-get install -y git

COPY ./app ./app

RUN poetry install --no-interaction --no-ansi

EXPOSE 8080

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8080
