# Review Sentiment Analysis - End-to-End MLOps

[![CI](https://github.com/harshitx11/flops_project/actions/workflows/ci.yml/badge.svg)](https://github.com/harshitx11/flops_project/actions/workflows/ci.yml)

This project classifies short review sentences (from products, restaurants, or movies) as **positive** or **negative**. It trains a scikit-learn Pipeline (TF-IDF followed by Logistic Regression) with fixed hyperparameters, serves it through FastAPI with a plain HTML/CSS/JavaScript web page, and includes tests, CI, a multi-stage Docker build that trains the model itself, Docker Compose, Prometheus, Grafana, and an optional Kubernetes manifest.

## Architecture

```text
UCI archive -> download/combine/validate -> stratified split -> fit fixed pipeline -> evaluate -> accept (macro F1 gate) -> artifact
                                                                                                                              |
web page -> FastAPI /predict -> loaded scikit-learn Pipeline (+ per-word contributions) -------------------------------------+
                     /metrics -> Prometheus -> Grafana
```

The test set is held out before fitting. There is **no hyperparameter search**: the pipeline settings are fixed and documented in `src/sentiment_mlops/models/train.py` (`MODEL_PARAMS`). Offline quality metrics come only from labeled test data. Production request metrics are operational and do not measure production accuracy.

## Dataset

The data is the [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) dataset (Kotzias et al., 2015, CC BY 4.0). It has 3,000 sentences, 1,000 each from Amazon product reviews, Yelp restaurant reviews, and IMDB movie reviews, labelled `1` (positive) or `0` (negative), with exactly 500 of each label per source.

`uv run download-data` downloads the zip, reads the three tab-separated files, and writes one combined file, `data/sentiment_sentences.csv`, with the columns `source,text,label`. `prepare_data.py` then maps labels to `positive`/`negative`, drops blank rows, removes exact duplicate sentences (18 in the current release), rejects any sentence that has conflicting labels, and checks that both classes are present. The data is fetched at setup time and is not committed to the repository.

## Model

| Step | Setting |
|---|---|
| `TfidfVectorizer` | unigrams + bigrams (`ngram_range=(1, 2)`), `sublinear_tf=True`, lowercase, Unicode accent stripping, no stop-word removal (so words like "not" are kept) |
| `LogisticRegression` | `C=1.0`, `max_iter=1000`, `random_state=42` |
| Split | 80/20 stratified, seed 42 |
| Acceptance gate | macro F1 ≥ `SENTIMENT_MIN_ACCEPTED_F1` (default 0.75) |

Reference run: 2,982 records after deduplication (2,385 for training, 597 for testing), **accuracy 0.792, macro F1 0.792**, and about 0.1 s of training time. For each prediction, the API also returns the words and phrases that pushed it most toward positive or negative. Each one's contribution is its TF-IDF weight × the Logistic Regression coefficient.

## Repository map

```text
src/sentiment_mlops/
  config.py              environment-driven settings (prefix SENTIMENT_)
  data/                  download_data.py, prepare_data.py
  models/                train.py, evaluate.py, predict.py
  api/app.py             FastAPI app, Prometheus metrics, static frontend
frontend/                index.html, style.css, script.js, config.js
tests/                   fixture-based unit and API tests (no network)
monitoring/              Prometheus scrape config, Grafana datasource + dashboard
deployment/              nginx reverse-proxy example, optional Kubernetes manifest
.github/workflows/       CI (lint, test, Docker smoke test) and monthly train/evaluate
artifacts/ reports/      model and evaluation outputs (gitignored)
```

## Run it locally (without Docker)

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs Python 3.11 automatically from `.python-version`.

From the repository root:

```bash
uv sync --extra dev
cp .env.example .env
uv run download-data
uv run train-model
uv run pytest
uv run ruff check .
uv run uvicorn sentiment_mlops.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. The same server provides `/predict`, `/model`, `/health`, `/ready`, `/metrics`, and the interactive API docs at `/docs`.

`download-data` reuses a valid local copy; pass `--force` to download it again. `train-model` writes `artifacts/sentiment_pipeline.joblib`, `artifacts/model_metadata.json`, and `reports/latest_evaluation.json`. A candidate below the macro F1 threshold is rejected: its report is still written, but it does not replace the current model.

## Run it with Docker (recommended)

Prerequisites: Docker Desktop (or Docker Engine with the Compose plugin) and internet access during the build.

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

You do **not** need to train on your machine first. The `Dockerfile` is multi-stage:

1. `base`: Python 3.11 slim with the locked runtime dependencies (installed by uv and cached).
2. `trainer`: downloads the dataset and trains the model. If the model fails the acceptance gate, the build fails.
3. `runtime`: only the source code, the frontend, and the accepted model artifact. It runs as a non-root user, and its health check calls `/ready`.

| Service | URL |
|---|---|
| Web page + API | <http://127.0.0.1:8000> |
| API docs | <http://127.0.0.1:8000/docs> |
| Prometheus | <http://127.0.0.1:9090> |
| Grafana | <http://127.0.0.1:3000> (login from `.env`, dashboard "Review Sentiment MLOps") |

All ports bind to loopback only. Useful commands:

```bash
docker compose logs -f api
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"text":"I love this phone"}'
docker compose down
```

`docker compose down -v` also deletes the Prometheus and Grafana volumes. To retrain, run `docker compose build --no-cache api`, then `docker compose up -d`.

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Body `{"text": "..."}`. Returns `predicted_class`, `label`, `positive_probability`, `positive_terms`, `negative_terms`, `model_version` |
| `GET /model` | Accepted model card: parameters, dataset counts, offline metrics |
| `GET /health` | Liveness, and whether a model is loaded |
| `GET /ready` | 200 when the model is loaded, otherwise 503 |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | OpenAPI UI |

Blank text returns 422. Text longer than `SENTIMENT_MAX_TEXT_LENGTH` returns 413. The API never logs submitted text.

## Metrics and monitoring

The API exports these metrics:

- `sentiment_http_requests_total{endpoint,status}`
- `sentiment_http_request_duration_seconds`
- `sentiment_predictions_total{predicted_class}`
- `sentiment_prediction_confidence` (a histogram of the predicted-class probability)
- `sentiment_prediction_failures_total`
- `sentiment_model_loaded`
- `sentiment_model_info{version}`
- `sentiment_model_eval_*`: accuracy, precision/recall/F1 for the positive class, F1 for the negative class, and macro F1

The evaluation gauges are read from `artifacts/model_metadata.json`, so a rejected run can never overwrite them. The Grafana dashboard is provisioned automatically.

## Configuration

Settings use the `SENTIMENT_` prefix and can come from environment variables or `.env`:

`SENTIMENT_DATA_DIR`, `SENTIMENT_ARTIFACT_DIR`, `SENTIMENT_REPORT_DIR`, `SENTIMENT_MODEL_VERSION`, `SENTIMENT_RANDOM_SEED`, `SENTIMENT_MIN_ACCEPTED_F1`, `SENTIMENT_MAX_TEXT_LENGTH`, `SENTIMENT_CORS_ORIGINS`, `SENTIMENT_MODEL_PATH`

In Docker, the acceptance threshold is a build argument (`docker compose build --build-arg SENTIMENT_MIN_ACCEPTED_F1=0.8`), because training happens at build time. Never put credentials in `.env.example`, frontend files, images, or metric labels.

## CI and training workflow

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`. It runs Ruff and pytest (using fixtures only, with no network), builds the runtime image, and smoke-tests `/ready` and `/predict` inside the container. `.github/workflows/train-evaluate.yml` runs manually or monthly: it downloads the data, trains, applies the gate, and uploads the evaluation JSON. It does not deploy anything.

## Deployment notes

- **VM**: install Docker, clone the repository, `cp .env.example .env`, set a strong `GRAFANA_ADMIN_PASSWORD` and `SENTIMENT_CORS_ORIGINS`, then run `docker compose up --build -d`. Put Nginx (see `deployment/nginx.conf`) or a load balancer in front for TLS.
- **Static frontend elsewhere** (for example Cloudflare Pages): publish `frontend/`, set `apiBaseUrl` in `frontend/config.js` to the public API origin, and add that site's origin to `SENTIMENT_CORS_ORIGINS`.
- **Kubernetes (optional)**: `docker build --target runtime -t YOUR_REGISTRY/sentiment-mlops:1.0.0 .`, push the image, update the image name in `deployment/kubernetes/deployment.yaml`, then run `kubectl apply -f deployment/kubernetes/deployment.yaml`.

## Troubleshooting and limitations

- **`/health` says `degraded`:** no model artifact was found. Run `uv run download-data && uv run train-model`, or rebuild the Docker image.
- **The Docker build fails in the `trainer` stage:** either the UCI download failed (check your internet connection) or the model missed the acceptance gate. The build log shows which one.
- **The page loads but analysis fails:** do not open `index.html` as a `file://` URL. Use the server URL instead.
- **Known limits:** the model is trained on 3k short English sentences from three domains. Sarcasm, negation spread over long texts, slang, and other domains will reduce accuracy. There is no neutral class: borderline text lands near 50%. The project does not include drift detection or automatic redeployment.

`make setup | data | train | test | lint | api | docker-up | docker-down | docker-logs` are shortcuts for the commands above.
