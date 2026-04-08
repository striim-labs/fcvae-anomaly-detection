# FCVAE Penny Transaction Anomaly Detection

Real-time anomaly detection on hourly penny transaction frequencies using a Frequency-enhanced Conditional VAE (Wang et al., WWW 2024). The FCVAE uses FFT-based conditioning (GFM + LFM) to learn periodic patterns and detects anomalies via elevated negative log-likelihood at the masked last-point position.

## Project Structure

```
fcvae-anomaly-detection/
│
├── code/                              # Numbered workflow (start here)
│   ├── 0_verify_setup.py             # Verify environment, data, artifacts
│   ├── 1_data_exploration.ipynb      # EDA: penny patterns, hourly cycles, FFT
│   ├── 2_model_design.ipynb          # FCVAE architecture walkthrough + training demo
│   ├── 3_train_model.py              # Full training pipeline for Penny_All
│   ├── 4_evaluate_model.py           # Evaluation: NLL distributions, metrics, plots
│   ├── 5_streaming_app.py            # FastAPI REST scoring service
│   └── 6_optimize.py                 # Hyperparameter grid search
│
├── src/                               # Reusable library code
│   ├── model.py                       # FCVAE architecture + attention modules
│   ├── scorer.py                      # NLL scoring, threshold calibration
│   ├── preprocess.py                  # Data loading, windowing, normalization
│   ├── train.py                       # Training utilities, augmentation
│   ├── schemas.py                     # Pydantic request/response models
│   └── utils.py                       # Device selection
│
├── data/                              # Dataset
│   ├── synthetic_transactions.csv     # 60-day synthetic dataset
│   └── generate_transactions.py       # Data generation script
│
├── models/fcvae/                      # Pre-trained artifacts
│   └── Penny_All/
│       ├── model.pt                   # Trained FCVAE weights
│       ├── scaler.pkl                 # StandardScaler for penny counts
│       └── scorer.pkl                 # FCVAEScorer with calibrated threshold
│
├── striim/                            # Striim platform integration
│   ├── fcvae-score-caller/            # Java Open Processor
│   ├── retrain-event-parser/          # Retrain event parser
│   └── types-src/                     # Striim type definitions
│
├── plots/                             # Generated visualizations
├── Dockerfile                         # Lightweight container for scoring API
├── docker-compose.yml                 # Scoring API service
├── pyproject.toml                     # All dependencies
├── STRIIM_PENNY.md                    # Striim penny pipeline setup
└── TECHNICAL.md                       # Architecture reference
```

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv): `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Quick Start

```bash
# Install dependencies
uv sync

# Verify setup
uv run python code/0_verify_setup.py

# Explore notebooks
uv run jupyter notebook code/
```

| Notebook | What You'll Learn |
|----------|-------------------|
| `1_data_exploration.ipynb` | Penny transaction patterns, hourly cycles, FFT analysis, anomaly periods |
| `2_model_design.ipynb` | FCVAE architecture, frequency conditioning, last-point masking, training demo |

## Scoring API

```bash
# Local
uv run python code/5_streaming_app.py

# Or via Docker
docker compose up --build
```

```bash
# Health check
curl http://localhost:8000/health

# Score a 24-hour window
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"values": [10,12,8,15,20,25,30,35,40,38,32,28,22,18,15,12,10,8,6,5,4,3,2,1]}'
```

## Workflow

| Step | File | Purpose |
|------|------|---------|
| 0 | `code/0_verify_setup.py` | Verify environment and artifacts |
| 1 | `code/1_data_exploration.ipynb` | Understand penny transaction data |
| 2 | `code/2_model_design.ipynb` | Understand FCVAE architecture, train, and evaluate |
| 5 | `code/5_streaming_app.py` | Deploy as REST API |
| 6 | `code/6_optimize.py` | Tune hyperparameters |

> **Note:** Do not run `code/3_train_model.py` or `code/4_evaluate_model.py` directly. Training and evaluation are demonstrated interactively in `code/2_model_design.ipynb`, which walks through the full pipeline with visualizations. The standalone scripts exist for automated/CI use only.
