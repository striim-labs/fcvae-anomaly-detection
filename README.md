# FCVAE Transaction Anomaly Detection

Real-time anomaly detection on hourly transaction frequencies using a Frequency-enhanced Conditional VAE (Wang et al., WWW 2024). The FCVAE uses FFT-based conditioning (GFM + LFM) to learn periodic patterns and detects anomalies via elevated negative log-likelihood at the masked last-point position.

Two complementary use cases demonstrate the architecture's versatility:

1. **Combo volume anomaly detection** — 4 independent models detect both *spikes* and *dips* in transaction volume across different network/transaction-type combinations, each with distinct periodicities (24h, 12h, 8h cycles).

2. **Penny transaction carding detection** — 1 pooled model detects carding attacks by identifying anomalous spikes in penny transaction (< $1) frequency.

## Project Structure

```
fcvae-anomaly-detection/
│
├── code/                              # Numbered workflow (start here)
│   ├── 0_verify_setup.py             # Verify environment, data, artifacts
│   ├── 1_combo_data_exploration.ipynb   # EDA: 4 combo time series, periodicities, FFT, anomalies
│   ├── 2_combo_anomaly_detection.ipynb  # Combo volume: spikes, dips, ramps across 4 models
│   ├── 3_penny_data_exploration.ipynb   # EDA: penny patterns, hourly cycles, FFT
│   ├── 4_penny_model_design.ipynb       # FCVAE architecture walkthrough + training demo
│   ├── 5_train_model.py              # Training pipeline (penny or combo mode)
│   ├── 6_evaluate_model.py           # Evaluation: NLL distributions, metrics, plots
│   ├── 7_streaming_app.py            # FastAPI REST scoring service (all 5 models)
│   └── 8_optimize.py                 # Hyperparameter grid search
│
├── src/                               # Reusable library code
│   ├── model.py                       # FCVAE architecture + attention modules
│   ├── scorer.py                      # NLL scoring, threshold calibration
│   ├── preprocess.py                  # Data loading (penny + combo), windowing, normalization
│   ├── train.py                       # Training utilities, augmentation
│   ├── schemas.py                     # Pydantic request/response models
│   └── utils.py                       # Device selection
│
├── data/                              # Dataset
│   ├── synthetic_transactions.csv     # 60-day synthetic dataset (train/val/test)
│   ├── synthetic_transactions_phase2.csv  # 10-day test-only evaluation set
│   └── generate_transactions.py       # Data generation script
│
├── models/fcvae/                      # Pre-trained artifacts
│   ├── Accel_CMP/                     # Accel/CMP combo model (24h cycle)
│   ├── Accel_nopin/                   # Accel/no-pin combo model (12h cycle)
│   ├── Star_CMP/                      # Star/CMP combo model (24h cycle)
│   ├── Star_nopin/                    # Star/no-pin combo model (8h cycle)
│   └── Penny_All/                     # Pooled penny transaction model
│
├── striim/                            # Striim platform integration
├── Dockerfile                         # Container for scoring API
├── docker-compose.yml                 # Scoring API service
├── pyproject.toml                     # All dependencies
├── STRIIM_FCVAE.md                    # Striim FCVAE pipeline setup
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
| `1_combo_data_exploration.ipynb` | 4 combo time series, diverse periodicities (24h/12h/8h), FFT analysis, spike/dip anomalies |
| `2_combo_anomaly_detection.ipynb` | 4-combo volume models detecting spikes, dips, and ramps with pre-trained models |
| `3_penny_data_exploration.ipynb` | Penny transaction patterns, hourly cycles, FFT analysis, anomaly periods |
| `4_penny_model_design.ipynb` | FCVAE architecture, frequency conditioning, last-point masking, training demo |

> **Notebooks 1 and 2 are sufficient for a full understanding** of the FCVAE approach. Notebooks 3 and 4 apply the same architecture to a different use case (penny carding detection) and are optional.

## Scoring API

```bash
# Recommended: run locally (faster startup, no build step)
uv run python code/7_streaming_app.py

# Or via Docker (slower — builds a container image first)
docker compose up --build
```

```bash
# Health check
curl http://localhost:8000/health

# Score penny transactions (default)
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"values": [10,12,8,15,20,25,30,35,40,38,32,28,22,18,15,12,10,8,6,5,4,3,2,1]}'

# Score a combo model
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"values": [1800,1600,1400,1200,1100,1300,1700,2000,2200,2400,2600,2800,3000,2900,2700,2500,2300,2100,1900,1700,1500,1300,1100,1000], "combo": "Accel_CMP"}'
```

## Workflow

| Step | File | Purpose |
|------|------|---------|
| 0 | `code/0_verify_setup.py` | Verify environment and artifacts |
| 1 | `code/1_combo_data_exploration.ipynb` | Explore combo transaction volume data |
| 2 | `code/2_combo_anomaly_detection.ipynb` | Combo volume anomaly detection (spikes + dips) |
| 3 | `code/3_penny_data_exploration.ipynb` | Explore penny transaction data |
| 4 | `code/4_penny_model_design.ipynb` | Understand FCVAE architecture, train, and evaluate |
| 7 | `code/7_streaming_app.py` | Deploy as REST API (all 5 models) |
| 8 | `code/8_optimize.py` | Tune hyperparameters |

> **Note:** Do not run `code/5_train_model.py` or `code/6_evaluate_model.py` directly. Training and evaluation are demonstrated interactively in the notebooks. The standalone scripts exist for automated/CI use and support `--mode penny` (default) or `--mode combo`.

## Further Reading

- **[TECHNICAL.md](TECHNICAL.md)** — FCVAE architecture reference: model equations, frequency conditioning (GFM/LFM), last-point masking, NLL scoring, KL annealing, and data augmentation details.
- **[STRIIM_FCVAE.md](STRIIM_FCVAE.md)** — Step-by-step guide for deploying the FCVAE scoring pipeline on Striim Platform, including the Open Processor, typed streams, and the TQL application definition.
