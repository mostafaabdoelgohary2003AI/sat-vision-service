# sat-vision-service

Satellite land-use classification as a deployable service: ResNet18 transfer learning on
real EuroSAT imagery, a FastAPI inference API with an upload web UI and a Postgres
prediction log. Dockerized and deploy-ready.

Built with AI pair-programming (Claude). All numbers below are real outputs of the
commands shown, not targets.

## Dataset
EuroSAT RGB (Sentinel-2, 64x64 tiles, 10 land-use classes). This repo uses the official
test + validation splits mirrored at huggingface.co/datasets/blanchon/EuroSAT_RGB:
10,800 real images (8,639 train / 2,161 held-out test, stratified). Download the shards with `python scripts/download_data.py` (mirrored at huggingface.co/datasets/blanchon/EuroSAT_RGB).

## Real results (held-out, 4 epochs fine-tuning the classifier head of ResNet18, CPU)
- test accuracy: 83.9% (10-class; chance = 10%)
- per-class recall: Residential 0.94, AnnualCrop 0.91, SeaLake 0.90, Forest 0.89,
  Industrial 0.89, Pasture 0.84, HerbaceousVegetation 0.83, River 0.77,
  PermanentCrop 0.70, Highway 0.67
- training time: ~52 s/epoch on CPU
- full history + confusion matrix: results/metrics.json

## Run locally (no Docker)
    pip install -r requirements.txt
    python scripts/download_data.py
    python -m src.train            # trains, evaluates, writes results/
    pytest -q                      # 3 tests
    uvicorn src.api:app --port 8000
    # open http://localhost:8000/ for the upload UI

## Run with Docker (Postgres included)
    docker compose up --build     # api on :8000, postgres 16 with migrations applied

Without Docker the API logs predictions to SQLite (`data/app.db`); with compose it uses
PostgreSQL (`DATABASE_URL`).

## API
    POST /predict   image file -> top-3 classes with probabilities (logged)
    GET  /history   latest logged predictions
    GET  /metrics   training metrics
    GET  /health

## Honest limits
- Only the classifier head is fine-tuned (frozen backbone) on CPU; unfreezing later layers
  or more epochs would raise accuracy further. Published EuroSAT results reach ~98% with
  full fine-tuning - this repo reports what it actually ran.
- 64x64 tiles are small; production satellite pipelines usually work on larger scenes.

## Stack
Python, PyTorch (ResNet18), FastAPI, vanilla-JS upload UI, PostgreSQL (Docker) / SQLite,
pytest, GitHub Actions CI.


## Results

Held-out accuracy: **94.1%** on 2,160 real test tiles (10-class EuroSAT; head-only baseline 83.9%). Per-class recall and full details: RUN-RESULTS.md.

