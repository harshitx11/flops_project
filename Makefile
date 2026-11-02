.PHONY: setup data train test lint api docker-up docker-down docker-logs
setup:
	uv sync --extra dev
data:
	uv run download-data
train:
	uv run train-model
test:
	uv run pytest
lint:
	uv run ruff check .
api:
	uv run uvicorn sentiment_mlops.api.app:app --app-dir src --host 127.0.0.1 --port 8000
docker-up:
	docker compose up --build -d
docker-down:
	docker compose down
docker-logs:
	docker compose logs -f api
