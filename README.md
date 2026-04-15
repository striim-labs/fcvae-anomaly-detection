# Striim AI Prototype: FCVAE Transaction Anomaly Detection

Real-time anomaly detection on hourly transaction frequencies using a Frequency-enhanced Conditional VAE (FCVAE) from Wang et al. (WWW 2024). The FCVAE uses FFT-based conditioning modules (GFM and LFM) to learn periodic patterns in time series and detects anomalies via elevated negative log-likelihood at the masked last-point position.

The repo demonstrates two complementary use cases that show the architecture's versatility. The **combo volume** use case trains 4 independent models that detect both spikes and dips in transaction volume across different network/transaction-type combinations -- for example, Visa vs Amex-style networks and credit (CMP) vs debit (no-pin) transaction types -- each with a distinct periodicity (24h, 12h, 8h cycles). The **penny carding** use case trains a single pooled model that detects carding attacks by identifying anomalous spikes in penny transaction (< $1) frequency. The numbered scripts walk through the penny path end-to-end; the combo path is covered in the notebooks and is well worth exploring to see how the same architecture handles multiple models with varying data patterns and anomaly types (spikes, dips, ramps).

The repo includes interactive notebooks for learning, production-grade source code, prebuilt reference artifacts for all 5 models, and a Dockerized scoring API that serves both use cases.

## Project Structure

```
fcvae-anomaly-detection/
├── code/                                # Numbered scripts -- canonical workflow
│   ├── 0_verify_setup.py               # Env + artifact check
│   ├── 1_train_model.py                # Train penny baseline -> models/fcvae/initial/Penny_All/
│   ├── 2_evaluate_model.py             # Evaluate saved artifacts (default: initial/Penny_All)
│   ├── 3_streaming_app.py              # FastAPI scoring service (Docker entrypoint)
│   ├── 4_grid_sweep.py                 # Sweep + retrain winner -> models/fcvae/best/Penny_All/
│   └── 5_export_onnx.py               # Export models to ONNX + parity validation
│
├── notebooks/                           # Interactive walkthroughs
│   ├── combo_data_exploration.ipynb     # 4 combo time series, periodicities, FFT, anomalies
│   ├── combo_anomaly_detection.ipynb    # Combo spike + dip detection with 4 prebuilt models
│   ├── penny_data_exploration.ipynb     # Penny patterns, hourly cycles
│   └── penny_model_design.ipynb        # FCVAE architecture walkthrough + training demo
│
├── src/                                 # Reusable library code
│   ├── model.py                         # FCVAE architecture + attention modules
│   ├── scorer.py                        # NLL scoring, threshold calibration
│   ├── preprocess.py                    # Data loading (penny + combo), windowing, normalization
│   ├── train.py                         # Low-level training utilities, augmentation
│   ├── training.py                      # Shared train_model() + save_training_artifacts()
│   ├── onnx_export.py                   # ONNX inference wrapper + export utility
│   ├── schemas.py                       # Pydantic request/response models
│   └── utils.py                         # Device selection
│
├── models/fcvae/                        # Pre-trained artifacts (committed, NEVER overwritten)
│   ├── Penny_All/                       # Pooled penny transaction model
│   ├── Accel_CMP/                       # Accel/CMP combo model (24h cycle)
│   ├── Accel_nopin/                     # Accel/no-pin combo model (12h cycle)
│   ├── Star_CMP/                        # Star/CMP combo model (24h cycle)
│   ├── Star_nopin/                      # Star/no-pin combo model (8h cycle)
│   ├── initial/Penny_All/               # gitignored -- 1_train_model.py output
│   └── best/Penny_All/                  # gitignored -- 4_grid_sweep.py output
│
├── data/                                # Dataset
│   ├── synthetic_transactions.csv       # 60-day synthetic dataset (train/val/test)
│   ├── synthetic_transactions_phase2.csv  # 10-day test-only evaluation set
│   └── generate_transactions.py         # Data generation script
│
├── striim/                              # Striim platform integration
├── Dockerfile                           # Container for scoring API
├── docker-compose.yml                   # Scoring API service
├── pyproject.toml                       # All dependencies
├── STRIIM.md                            # Striim FCVAE pipeline setup
└── TECHNICAL.md                         # Architecture reference
```

The numbered scripts in `code/` are the first-class reproduction path. Notebooks in `notebooks/` provide the deeper explanations behind each design decision. User-trained output goes to gitignored `models/fcvae/initial/` and `models/fcvae/best/` directories so the committed prebuilt artifacts are never overwritten.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker (optional, for the scoring API)
- ONNX dependencies (optional, for step 5): `uv sync --extra onnx`

## Going through the code

### 1. Install dependencies

```bash
uv sync
```

### 2. Train the baseline

```bash
uv run python code/1_train_model.py
```

Trains a penny FCVAE baseline with conservative defaults (latent_dim=4, 15 epochs, no augmentation). Output goes to `models/fcvae/initial/Penny_All/`.

### 3. Evaluate the baseline

```bash
uv run python code/2_evaluate_model.py
```

Reads `models/fcvae/initial/Penny_All/` by default and prints precision, recall, and F1 metrics.

> **Note:** `code/3_streaming_app.py` is intentionally skipped here -- it is the Docker entrypoint for the scoring API. See the [Docker demo](#docker-demo-scoring-api) section below.

### 4. Grid sweep to find a better config

```bash
uv run python code/4_grid_sweep.py
```

Sweeps ~11 configurations over latent_dim, KL warmup, learning rate, and augmentation. After the sweep, the script automatically retrains the winning configuration end-to-end and saves artifacts to `models/fcvae/best/Penny_All/`.

### 5. Evaluate the best-config model

```bash
uv run python code/2_evaluate_model.py --model-dir models/fcvae/best/Penny_All
```

Should match or closely approximate the prebuilt reference metrics.

### 6. Export to ONNX

```bash
uv sync --extra onnx
uv run python code/5_export_onnx.py
```

Exports all 5 prebuilt FCVAE models to ONNX format (opset 18, required for FFT ops) and validates parity against PyTorch using ONNX Runtime. Each model directory gets a `model.onnx` file and a `model_config.json` containing scaler parameters, anomaly thresholds, and ONNX metadata -- everything a downstream consumer (such as a JVM-based scorer using ONNX Runtime) needs to run inference without Python.

## Read through the notebooks

The notebooks cover both the penny and combo use cases in depth. Start with the combo notebooks -- they show how 4 independent FCVAE models handle time series with fundamentally different periodicities (24h, 12h, 8h cycles) and detect both **spikes** and **dips**, which demonstrates the architecture's adaptability far better than the single penny model alone.

| Notebook | What you'll learn |
|----------|---|
| `combo_data_exploration.ipynb` | 4 combo time series with diverse periodicities (24h/12h/8h), FFT analysis, spike/dip/ramp anomaly patterns |
| `combo_anomaly_detection.ipynb` | How the same FCVAE architecture detects both spikes AND dips across the 4 combo models using the prebuilts |
| `penny_data_exploration.ipynb` | Penny transaction patterns, hourly cycles, anomaly periods |
| `penny_model_design.ipynb` | FCVAE architecture walkthrough, frequency conditioning (GFM/LFM), last-point masking, training demo |

```bash
uv run jupyter notebook notebooks/
```

## Docker demo (scoring API)

`code/3_streaming_app.py` is a FastAPI scoring service that loads all 5 prebuilt models (penny + 4 combos) at startup. Running it directly with `uv run python code/3_streaming_app.py` is the fastest way to start the API locally. Alternatively, use Docker:

```bash
# Option 1: run directly (faster startup, no build step)
uv run python code/3_streaming_app.py

# Option 2: run via Docker
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
| 0 | `0_verify_setup.py` | Optional troubleshooting |
| 1 | `1_train_model.py` | Train penny baseline -> models/fcvae/initial/Penny_All/ |
| 2 | `2_evaluate_model.py` | Evaluate saved artifacts (default: initial/Penny_All) |
| 3 | `3_streaming_app.py` | FastAPI scoring service |
| 4 | `4_grid_sweep.py` | Sweep + retrain winner -> models/fcvae/best/Penny_All/ |
| 5 | `5_export_onnx.py` | Export models to ONNX + parity validation |

Prebuilt artifacts in `models/fcvae/Penny_All/`, `Accel_CMP/`, `Accel_nopin/`, `Star_CMP/`, and `Star_nopin/` are the reference and are never touched by any script. User output goes to gitignored `initial/` and `best/` subdirectories.

## Detection methodology

The FCVAE operates on 24-point sliding windows over hourly transaction frequencies. Each window is conditioned on frequency-domain features extracted via FFT: the Global Frequency Module (GFM) applies FFT to the entire window (excluding the last point) to capture the dominant periodicity, while the Local Frequency Module (LFM) splits the window into overlapping 8-hour sub-windows, applies FFT to each, and uses multi-head self-attention to capture shorter-range trends.

During inference, position [-1] (the last hour) is masked from both frequency modules, making it the only genuine prediction. The anomaly score is the negative log-likelihood (NLL) under the decoder's Gaussian output distribution at that masked position. Lower NLL means more anomalous. A threshold is calibrated via F1-max on labeled data, and points scoring below the threshold are flagged as anomalies.

## Further Reading

- **[TECHNICAL.md](TECHNICAL.md)** -- FCVAE architecture reference: model equations, frequency conditioning (GFM/LFM), last-point masking, NLL scoring, KL annealing, and data augmentation details.

- **[STRIIM.md](STRIIM.md)** -- The FCVAE scoring service is designed to run inside Striim Platform as a real-time pipeline component. STRIIM.md walks through the full integration: packaging the scoring API as an Open Processor module, defining typed streams for the input transaction events and output anomaly scores, wiring the TQL application that connects a Kafka source to the FCVAE scorer and routes results to downstream sinks. This demonstrates that models of this complexity -- a multi-model VAE with FFT conditioning, per-window NLL scoring, and calibrated thresholds -- can be deployed as a first-class operator inside Striim's continuous processing engine, scoring transactions in real time alongside the rest of the platform's ingestion and transformation pipeline.
