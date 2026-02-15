# FCVAE Anomaly Detection — Technical Reference

Detailed architecture and internals of the FCVAE (Frequency-enhanced Conditional Variational Autoencoder) anomaly detection system. For setup instructions and usage, see [README.md](README.md).

---

## Table of Contents

1. [Core Model Architecture](#1-core-model-architecture)
2. [Attention Mechanisms](#2-attention-mechanisms)
3. [Scoring System](#3-scoring-system)
4. [Registry System](#4-registry-system)
5. [Data Augmentation](#5-data-augmentation)
6. [Streaming Detector](#6-streaming-detector)
7. [Training Pipeline](#7-training-pipeline)
8. [Docker Infrastructure](#8-docker-infrastructure)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Core Model Architecture

### 1.1 FCVAEConfig

Configuration dataclass for model hyperparameters:

```python
@dataclass
class FCVAEConfig:
    # Window and latent dimensions
    window: int = 24                      # Hours per window (daily)
    latent_dim: int = 8                   # Latent space dimension
    condition_emb_dim: int = 16           # Frequency condition embedding

    # Attention parameters
    d_model: int = 64                     # Attention model dimension
    d_inner: int = 128                    # FFN inner dimension
    n_head: int = 4                       # Number of attention heads

    # Local Frequency Module (LFM)
    kernel_size: int = 8                  # LFM sub-window size (hours)
    stride: int = 4                       # LFM stride → 5 sub-windows

    # Architecture
    dropout_rate: float = 0.05            # Dropout rate
    hidden_dims: Tuple[int, ...] = (64, 64)  # Encoder/decoder hidden layers

    # MCMC scoring (test time)
    mcmc_iterations: int = 10             # MCMC refinement steps
    mcmc_samples: int = 64                # Samples for averaging
    mcmc_rate: float = 0.2                # Threshold percentile
    mcmc_mode: int = 2                    # Replace last point only
```

### 1.2 FCVAE Architecture

The FCVAE combines two frequency conditioning modules with a VAE:

```
Input: x (B, 1, 24)  [Hourly counts for 24 hours]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│                 FREQUENCY CONDITIONING                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────────────┐  ┌─────────────────────────┐ │
│  │ Global Frequency    │  │ Local Frequency         │ │
│  │ Module (GFM)        │  │ Module (LFM)            │ │
│  │                     │  │                         │ │
│  │ FFT(entire window)  │  │ FFT(8h sub-windows)     │ │
│  │        ↓            │  │        ↓                │ │
│  │ emb_global (Linear) │  │ emb_local (Linear)      │ │
│  │        ↓            │  │        ↓                │ │
│  │ (B, 1, cond_dim)    │  │ Self-Attention (4 heads)│ │
│  │                     │  │        ↓                │ │
│  │                     │  │ Mean Pool               │ │
│  │                     │  │        ↓                │ │
│  │                     │  │ (B, 1, cond_dim)        │ │
│  └─────────────────────┘  └─────────────────────────┘ │
│            │                        │                  │
│            └───────┬────────────────┘                  │
│                    ▼                                   │
│         condition: (B, 1, 2*cond_dim)                  │
└────────────────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐    ┌─────────────────────────────┐
│    ENCODER      │    │         DECODER             │
├─────────────────┤    ├─────────────────────────────┤
│                 │    │                             │
│ [x, condition]  │    │ [z, condition]              │
│       ↓         │    │       ↓                     │
│ Linear(W+2C, 64)│    │ Linear(L+2C, 64)            │
│       ↓         │    │       ↓                     │
│ LeakyReLU       │    │ LeakyReLU                   │
│       ↓         │    │       ↓                     │
│ Linear(64, 64)  │    │ Linear(64, 64)              │
│       ↓         │    │       ↓                     │
│ LeakyReLU       │    │ LeakyReLU                   │
│       ↓         │    │       ↓                     │
│ ┌─────┴─────┐   │    │ Linear(64, W)               │
│ ↓           ↓   │    │       ↓                     │
│ fc_mu    fc_var │    │ ┌─────┴─────┐               │
│ (64→8)   (64→8) │    │ ↓           ↓               │
│ ↓           ↓   │    │ fc_mu_x  fc_var_x           │
│ μ       Softplus│    │ (W→W)    (W→W, Softplus)    │
│           ↓     │    │ ↓           ↓               │
│           σ²    │    │ μ_x        σ²_x             │
└─────────────────┘    └─────────────────────────────┘
         │                       ▲
         ▼                       │
┌─────────────────┐              │
│ Reparameterize  │              │
│ z = μ + ε·√σ²   │──────────────┘
│ (B, latent_dim) │
└─────────────────┘
```

### 1.3 Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_condition` | `(x: Tensor) → Tensor` | Compute frequency condition (GFM + LFM) |
| `encode` | `(x: Tensor) → (μ, σ²)` | Map input + condition to latent distribution |
| `decode` | `(z: Tensor) → (μ_x, σ²_x)` | Map latent + condition to output distribution |
| `reparameterize` | `(μ, σ²) → z` | Sample latent using reparameterization trick |
| `forward` | `(x, mode, kl_weight) → (...)` | Full forward pass returning loss |
| `loss_func` | `(μ_x, σ²_x, x, μ, σ², kl_weight) → loss` | ELBO loss computation |
| `score_single_pass` | `(x, n_samples) → scores` | Fast NLL scoring (streaming) |
| `score_mcmc` | `(x) → (refined, scores)` | Accurate MCMC scoring (evaluation) |
| `reconstruct` | `(x) → (μ_x, σ²_x)` | Get reconstruction for visualization |

### 1.4 Last-Point Masking

The FCVAE is a **point-level detector**: each 24-hour window exists to predict the **last point only** (position [-1]). Both frequency conditioning modules explicitly exclude position [-1]:

- **GFM**: Computes `rfft(x[:,:,:-1])` — the last point is excluded from the FFT
- **LFM**: Zeros the last point via `x_l[:,:,:,-1] = 0` before local frequency extraction

This means position [-1] is the only **genuine prediction** — positions [0..22] are reconstruction scores where the model had access to the values through the conditioning signal.

### 1.5 Loss Function

**ELBO (Evidence Lower Bound):**

```
Loss = Reconstruction_Loss + kl_weight × KLD_Loss

Reconstruction_Loss = -log p(x|z) = 0.5 × Σ[log(σ²_x) + (x - μ_x)²/σ²_x]

KLD_Loss = KL(q(z|x) || p(z)) = 0.5 × Σ[μ² + σ² - log(σ²) - 1]
```

Where `p(z) = N(0, I)` is the prior.

---

## 2. Attention Mechanisms

Located in `attention.py`, these modules implement the Local Frequency Module's self-attention.

### 2.1 EncoderLayer_selfattn

Combines multi-head self-attention with position-wise feed-forward network:

```python
class EncoderLayer_selfattn(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        enc_output, attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn
```

### 2.2 MultiHeadAttention

Standard transformer multi-head attention with residual connection and layer normalization:

```
Input: Q, K, V (all from same input for self-attention)
  │
  ├──► Linear(d_model → n_head × d_k) → Q'
  ├──► Linear(d_model → n_head × d_k) → K'
  └──► Linear(d_model → n_head × d_v) → V'
         │
         ▼
  ScaledDotProductAttention(Q', K', V')
         │
         ▼
  Concat heads → Linear(n_head × d_v → d_model)
         │
         ▼
  Residual + LayerNorm
         │
         ▼
  Output
```

### 2.3 ScaledDotProductAttention

Core attention operation:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

### 2.4 PositionwiseFeedForward

Two-layer feed-forward with residual:

```
FFN(x) = ReLU(Conv1d(x)) → Conv1d → + x → LayerNorm
```

---

## 3. Scoring System

### 3.1 FCVAEScorerConfig

```python
@dataclass
class FCVAEScorerConfig:
    # Threshold configuration
    threshold_method: str = "percentile"    # "percentile" or "f1_max"
    threshold_percentile: float = 5.0       # LOW percentile (5th)
    hard_criterion_k: int = 3               # Points below threshold to flag

    # Scoring mode
    score_mode: str = "single_pass"         # "single_pass" or "mcmc"
    n_samples: int = 16                     # Latent samples for scoring

    # Window decision logic
    decision_mode: str = "count_only"       # See decision modes below
    severity_margin: float = 0.5            # Additional margin below threshold
    outlier_z_threshold: float = 3.0        # Z-score threshold
```

### 3.2 Critical Scoring Semantics

**IMPORTANT: Score direction is INVERTED from LSTM-AE:**

| Aspect | LSTM-AE | FCVAE |
|--------|---------|-------|
| Score type | Mahalanobis distance | Negative log-likelihood (NLL) |
| Normal points | Low score | High (less negative) score |
| Anomalous points | High score | Low (more negative) score |
| Threshold comparison | `score > threshold` | `score < threshold` |
| Percentile for threshold | High (95th) | **Low (5th)** |

### 3.3 Last-Point Scoring in Streaming

In the final streaming application, the detector uses **last-point-only scoring**:

- Each hourly window produces a single score: `point_scores[-1]` (the NLL at position [-1])
- The anomaly decision is: `last_point_score < last_point_threshold`
- The threshold is calibrated specifically on the distribution of last-point scores from validation data
- This aligns with the model design — position [-1] is the only position masked from frequency conditioning

The legacy `count_only`/`severity`/`hybrid` modes (which count how many of the 24 positions exceed threshold) are retained for backward compatibility but are not used in the default streaming configuration.

### 3.4 Threshold Methods

| Method | Description |
|--------|-------------|
| `percentile` | Use 5th percentile of normal validation scores |
| `f1_max` | Search for threshold maximizing F1 score |
| `youden` | Maximize Youden's J = sensitivity + specificity - 1 |
| `midpoint` | Midpoint between normal and anomaly distributions |

### 3.5 Decision Modes

| Mode | Logic |
|------|-------|
| `last_point` | **(Default in streaming)** Flag if last-point score < last_point_threshold |
| `count_only` | Flag if >= k points below threshold |
| `k1` | Flag if any single point below threshold |
| `severity` | Flag if k-count OR any point < (threshold - margin) |
| `zscore` | Flag if k-count OR any point with z < -z_threshold |
| `hybrid` | Flag if k-count OR severity OR zscore |

---

## 4. Registry System

### 4.1 Combo Keys

Four independent models, one per network/transaction type combination:

```python
COMBO_KEYS = [
    ("Accel", "CMP"),      # Accelerator network, CMP transactions
    ("Accel", "no-pin"),   # Accelerator network, no-pin transactions
    ("Star", "CMP"),       # Star network, CMP transactions
    ("Star", "no-pin")     # Star network, no-pin transactions
]
```

### 4.2 FCVAERegistry

Central manager for all combo models:

```python
class FCVAERegistry:
    # Per-combo storage
    models: Dict[Combo, FCVAE]                    # Trained models
    scorers: Dict[Combo, FCVAEScorer]            # Fitted scorers
    scalers: Dict[Combo, StandardScaler]         # Data normalizers
    training_histories: Dict[Combo, Dict]        # Loss curves
    model_versions: Dict[Combo, int]             # Version tracking

    # Global configuration
    model_config: FCVAEConfig
    scorer_config: FCVAEScorerConfig
    augment_config: AugmentConfig
    device: torch.device
```

### 4.3 Key Registry Methods

| Method | Description |
|--------|-------------|
| `get_model(combo)` | Get or create model for combo |
| `get_scorer(combo)` | Get or create scorer for combo |
| `train_combo(combo, ...)` | Train model with augmentation and KL annealing |
| `fit_scorer(combo, val_loader)` | Fit scorer using percentile threshold |
| `fit_scorer_with_calibration(...)` | Calibrate using synthetic anomalies |
| `predict(combo, data_loader)` | Generate predictions and scores |
| `save_all(output_dir)` | Save all artifacts to disk |
| `load_all(model_dir)` | Load all artifacts from disk |

### 4.4 Model Directory Structure

```
models/fcvae/
├── registry_config.pkl              # FCVAEConfig, FCVAEScorerConfig, AugmentConfig
├── oracle_thresholds.json           # Per-combo last-point thresholds (F1-calibrated)
├── Accel_CMP/
│   ├── model.pt                     # {"model_config": FCVAEConfig,
│   │                                #  "model_state_dict": OrderedDict}
│   ├── scorer.pkl                   # Pickled FCVAEScorer
│   ├── scaler.pkl                   # Pickled StandardScaler
│   └── history.pkl                  # {"train_loss": [...], "val_loss": [...],
│                                    #  "best_epoch": int, "learning_rates": [...],
│                                    #  "kl_weights": [...]}
├── Accel_nopin/
│   └── [same structure]
├── Star_CMP/
│   └── [same structure]
└── Star_nopin/
    └── [same structure]
```

---

## 5. Data Augmentation

### 5.1 AugmentConfig

```python
@dataclass
class AugmentConfig:
    missing_data_rate: float = 0.01    # 1% of points zeroed
    point_ano_rate: float = 0.05       # 5% of batch get point anomalies
    seg_ano_rate: float = 0.1          # 10% of batch get segment swaps
```

### 5.2 Augmentation Types

#### Point Anomaly (`point_ano`)

```
Original:  [v1, v2, v3, ..., v23, v24]
Augmented: [v1, v2, v3, ..., v23, v24 + noise]  where noise ∈ [-10, 10]
Label:     [0,  0,  0,  ..., 0,   1]            (only last point anomalous)
```

- Samples `rate × batch_size` windows
- Adds random noise to last point only
- Creates NEW samples (concatenated to batch)

#### Segment Anomaly (`seg_ano`)

```
Window A:  [a1, a2, ..., a12, a13, ..., a24]
Window B:  [b1, b2, ..., b12, b13, ..., b24]
                         ↓ swap ↓
Augmented A: [a1, a2, ..., a12, b13, ..., b24]
Augmented B: [b1, b2, ..., b12, a13, ..., a24]
Labels:      [0,  0,  ..., 0,   1,   ..., 1]   (swapped region is anomalous)
```

- Pairs random windows and swaps tail segments
- Segment start position: hours 6-18
- Creates NEW samples

#### Missing Data (`missing_data_injection`)

```
Original:  [v1, v2, v3, v4, v5, ...]
Augmented: [v1, 0,  v3, v4, 0,  ...]  (random ~1% of points zeroed)
```

- Modifies existing samples in-place
- Updates missing mask `z` for affected positions

### 5.3 Batch Augmentation Flow

```python
def batch_augment(x, y, z, config):
    """
    Args:
        x: (B, 1, W) input windows
        y: (B, W) labels
        z: (B, W) missing mask
    Returns:
        Augmented (x, y, z) with increased batch size
    """
    # 1. Point anomalies (creates new samples)
    x_point, y_point, z_point = point_ano(x, y, z, config.point_ano_rate)

    # 2. Segment anomalies (creates new samples)
    x_seg, y_seg, z_seg = seg_ano(x, y, z, config.seg_ano_rate)

    # 3. Missing data (modifies in-place)
    x, z = missing_data_injection(x, z, config.missing_data_rate)

    # Concatenate all
    x = torch.cat([x, x_point, x_seg], dim=0)
    y = torch.cat([y, y_point, y_seg], dim=0)
    z = torch.cat([z, z_point, z_seg], dim=0)

    return x, y, z
```

---

## 6. Streaming Detector

### 6.1 FCVAEStreamingDetector

Real-time detector implementing the `BaseDetector` interface:

```python
class FCVAEStreamingDetector(BaseDetector):
    def __init__(
        self,
        model_path: str = "models/fcvae",
        combo: Optional[Tuple[str, str]] = None,
        window_size: int = 24,
        min_samples: int = 24,
        n_samples: int = 16,
        device: Optional[str] = None,
        decision_mode: str = "last_point",
        oracle_threshold: Optional[float] = None,
    ):
        # Loads model.pt, scorer.pkl, scaler.pkl from disk
        ...
```

### 6.2 Last-Point Detection Flow

The streaming detector uses **last-point scoring** by default. Each new hourly sample triggers:

```python
def _compute_sequence_score_and_errors(self, sequence):
    # 1. Score all positions via MCMC mode 2
    _, point_scores = self.model.score_mcmc(x)  # (1, 1, 24) → (24,)

    # 2. Extract last-point score (the only genuine prediction)
    last_point_score = point_scores[-1]

    # 3. Compare against calibrated threshold
    threshold = self.scorer.last_point_threshold
    is_anomaly = last_point_score < threshold

    return is_anomaly, last_point_score, point_scores, ...
```

### 6.3 Key Differences from LSTM Detector

| Aspect | LSTM Detector | FCVAE Detector |
|--------|---------------|----------------|
| DoW features | Required (3 features per hour) | Not needed |
| Input shape | `(B, 24, 3)` with DoW | `(B, 1, 24)` normalized counts |
| Score computation | Mahalanobis distance | Single-pass NLL |
| Score direction | Higher = anomalous | **Lower = anomalous** |
| Threshold comparison | `>` | `<` |
| Window size | 336 hours (weekly) | 24 hours (daily) |
| Decision logic | k-of-W count | Last-point only |

### 6.4 Anomaly Localization

```python
def _localize_anomaly(
    self,
    point_predictions: np.ndarray,  # (24,) boolean
    reconstruction_errors: np.ndarray,  # (24,) float
    timestamps: List[str],
    scale_hours: int = 6
) -> Dict:
    """
    Find the scale_hours window with highest anomaly concentration.

    Returns:
        {
            "anomaly_start": str,      # Start timestamp
            "anomaly_end": str,        # End timestamp
            "scale_hours": int,        # Localization window size
            "contrast_ratio": float    # Inside/outside error ratio
        }
    """
```

---

## 7. Training Pipeline

### 7.1 Training Loop

```python
def train_combo(combo, train_loader, val_loader, epochs=30, ...):
    model = self.get_model(combo)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(epochs):
        # KL annealing: linear ramp from 0 to 1
        kl_weight = min(1.0, epoch / kl_warmup_epochs)

        # Training
        model.train()
        for batch in train_loader:
            x, y, z = batch
            x, y, z = batch_augment(x, y, z, self.augment_config)

            optimizer.zero_grad()
            _, _, _, _, _, loss = model(x, mode="train", kl_weight=kl_weight)
            loss.backward()
            clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = evaluate(model, val_loader, kl_weight)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        scheduler.step()
```

### 7.2 KL Annealing

Gradually increases KL divergence weight during training:

```
kl_weight = min(1.0, epoch / kl_warmup_epochs)

Epoch 0:  kl_weight = 0.0   (pure reconstruction)
Epoch 5:  kl_weight = 0.5   (balanced)
Epoch 10: kl_weight = 1.0   (full ELBO)
```

### 7.3 Threshold Calibration

**Method 1: Percentile (Simple)**
```python
def fit_scorer(combo, val_loader):
    scores = score_all_windows(val_loader)
    threshold = np.percentile(scores, 5.0)  # 5th percentile
```

**Method 2: F1-Max with Synthetic Anomalies (Balanced)**
```python
def fit_scorer_with_calibration(combo, val_loader, ...):
    # 1. Score normal windows
    normal_scores = score_windows(normal_data)

    # 2. Create synthetic anomalies (spike + dip injections)
    spike_data = inject_spike(normal_data, spike_hours, magnitude)
    dip_data = inject_dip(normal_data, dip_hours, magnitude)

    # 3. Score anomaly windows
    anomaly_scores = score_windows([spike_data, dip_data])

    # 4. Find F1-maximizing threshold
    threshold = find_optimal_threshold(normal_scores, anomaly_scores, method="f1_max")
```

### 7.4 Command-Line Interface

```bash
python train_fcvae.py \
    --data-path data/synthetic_transactions.csv \
    --output-dir models/fcvae \
    --window-size 24 \
    --stride 1 \
    --latent-dim 8 \
    --epochs 75 \
    --lr 5e-4 \
    --patience 5 \
    --batch-size 64 \
    --grad-clip 2.0 \
    --threshold-percentile 5.0 \
    --hard-criterion-k 3 \
    --kl-warmup-epochs 35
```

---

## 8. Docker Infrastructure

### 8.1 Service Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Compose Network                       │
│                     (streaming_network)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │  Zookeeper   │ :2181                                         │
│  │  (Kafka      │◄────────────────────────────────┐            │
│  │  Coordination)│                                 │            │
│  └──────────────┘                                 │            │
│         ▲                                         │            │
│         │                                         │            │
│  ┌──────┴───────┐     ┌─────────────┐            │            │
│  │    Kafka     │:9092│  Producer   │            │            │
│  │  (Message    │◄────┤  (Data      │            │            │
│  │   Broker)    │     │   Generator)│            │            │
│  └──────────────┘     └─────────────┘            │            │
│         │                                         │            │
│         │ Kafka Topic: anomaly_stream             │            │
│         ▼                                         │            │
│  ┌──────────────┐                                │            │
│  │     App      │:8050  ◄──────┐                 │            │
│  │  (Dash +     │       │      │                 │            │
│  │   FCVAE)     │       │      │                 │            │
│  └──────────────┘       │      │                 │            │
│         │               │      │                 │            │
│         │ Spark Submit  │      │ Health Check   │            │
│         ▼               │      │                 │            │
│  ┌──────────────┐       │      │                 │            │
│  │ Spark Master │:8080  │      │                 │            │
│  │              │:7077◄─┘      │                 │            │
│  └──────┬───────┘              │                 │            │
│         │                      │                 │            │
│         ▼                      │                 │            │
│  ┌──────────────┐              │                 │            │
│  │ Spark Worker │──────────────┘                 │            │
│  │  (2GB, 2core)│                                │            │
│  └──────────────┘                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Service Details

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| **zookeeper** | `confluentinc/cp-zookeeper:7.5.0` | 2181 | Kafka coordination |
| **kafka** | `confluentinc/cp-kafka:7.5.0` | 9092, 29092 | Message broker |
| **spark-master** | `apache/spark:3.5.3-python3` | 8080, 7077 | Spark coordination |
| **spark-worker** | `apache/spark:3.5.3-python3` | - | Spark execution |
| **producer** | Custom | - | Data streaming |
| **app** | Custom | 8050 | Dash + FCVAE |

### 8.3 Environment Variables

```yaml
app:
  environment:
    # Kafka connection
    KAFKA_BOOTSTRAP_SERVERS: kafka:29092
    KAFKA_TOPIC: anomaly_stream

    # Spark connection
    SPARK_MASTER: spark://spark-master:7077

    # FCVAE configuration
    DETECTOR_TYPE: fcvae
    WINDOW_SIZE: 24                  # Hours per window
    MIN_SAMPLES: 24                  # Min samples before detection
    MODEL_PATH: models/fcvae        # Model directory
    COMBO: Accel_CMP                # Which combo to load
    N_SAMPLES: 16                   # Latent samples for scoring
```

### 8.4 Volume Mounts

```yaml
volumes:
  - ./models:/app/models:ro     # Trained models (read-only)
  - ./data:/app/data:ro         # Data files (read-only)
```

---

## 9. Data Flow Diagrams

### 9.1 Training Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

synthetic_transactions.csv
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ TransactionPreprocessor.load_and_aggregate()                     │
│   - Parse timestamps                                             │
│   - Group by (network_type, transaction_type, hour)             │
│   - Count transactions per hour                                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
combo_hourly: Dict[Tuple[str, str], DataFrame]
   {"Accel_CMP": df, "Accel_nopin": df, "Star_CMP": df, "Star_nopin": df}
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ create_sliding_splits(window=24, stride=1)                       │
│   - Create overlapping 24-hour windows                          │
│   - Split: train (days 1-42), val (days 43-50), test (51-60)    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
splits: {"train": (N_train, 24), "val": (N_val, 24), "test": (N_test, 24)}
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ normalize_sliding_windows(fit_on="train")                        │
│   - Fit StandardScaler on training data                         │
│   - Transform all splits                                         │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
normalized_splits + scalers per combo
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ create_sliding_dataloaders(batch_size=64)                        │
│   - Wrap in SlidingWindowDataset                                │
│   - Create DataLoader with shuffle (train only)                 │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ FCVAERegistry.train_combo() [per combo]                          │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │ For each epoch:                                           │ │
│   │   1. Compute kl_weight = min(1.0, epoch / warmup_epochs)  │ │
│   │   2. For each batch:                                      │ │
│   │      - batch_augment(x, y, z) → augmented batch           │ │
│   │      - model.forward(x, kl_weight) → loss                 │ │
│   │      - loss.backward()                                    │ │
│   │      - clip_grad_value_(params, 2.0)                      │ │
│   │      - optimizer.step()                                   │ │
│   │   3. Validate and check early stopping                    │ │
│   │   4. scheduler.step()                                     │ │
│   └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ FCVAERegistry.fit_scorer_with_calibration() [per combo]          │
│   1. Score normal validation windows                             │
│   2. Inject synthetic spike/dip anomalies                       │
│   3. Score anomaly windows                                       │
│   4. Find F1-maximizing threshold                               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ FCVAERegistry.save_all(output_dir)                               │
│   - Save model.pt, scorer.pkl, scaler.pkl, history.pkl          │
│   - Save registry_config.pkl                                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
models/fcvae/{Accel_CMP, Accel_nopin, Star_CMP, Star_nopin}/
```

### 9.2 Inference Data Flow (Streaming)

```
┌─────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                         │
│                    (Last-Point Scoring)                           │
└─────────────────────────────────────────────────────────────────┘

Producer → Kafka Topic (anomaly_stream)
                    │
                    │ JSON: {"timestamp": "...", "value": 123,
                    │        "produced_at": "...", "sequence_id": N}
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Spark Structured Streaming                                       │
│   - Subscribe to Kafka topic                                     │
│   - Parse JSON messages                                          │
│   - Micro-batch every 1 second                                   │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ main.py: process_batch()                                         │
│   - Append to data_store["data"] (deque)                        │
│   - Track samples_since_last_window                             │
└─────────────────────────────────────────────────────────────────┘
                    │
                    │ For each new hourly sample (stride 1)
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ main.py: run_anomaly_detection() [background thread]             │
│   - Get last 24 samples from buffer                             │
│   - Call detector.score_window_detailed(df)                     │
│   - Check last_point_score < threshold → 1-hour anomaly         │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ FCVAEStreamingDetector.score_window_detailed(df)                 │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 1. NORMALIZE                                            │   │
│   │    values = df["value"].values.reshape(-1, 1)           │   │
│   │    normalized = scaler.transform(values).flatten()      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 2. SCORE (MCMC mode 2)                                  │   │
│   │    _, point_scores = model.score_mcmc(x)                │   │
│   │    → (1, 1, 24) NLL scores per hour                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 3. LAST-POINT DECISION                                  │   │
│   │    last_point_score = point_scores[-1]                  │   │
│   │    is_anomaly = last_point_score < last_point_threshold │   │
│   └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 4. RETURN                                               │   │
│   │    {is_anomaly, last_point_score, point_scores,         │   │
│   │     point_threshold, timestamps, values, ...}           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ main.py: update_dashboard() [Dash callback every 1s]             │
│   - Plot time series with anomaly markers (red = detected,      │
│     orange = PA-caught)                                          │
│   - Update status cards and tables                               │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
            Dash Dashboard (http://localhost:8050)
```

---

## 10. Configuration Reference

### 10.1 Model Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `window` | 24 | 12-48 | Hours per detection window |
| `latent_dim` | 8 | 4-32 | Latent space dimension |
| `condition_emb_dim` | 16 | 8-64 | Frequency embedding dimension |
| `d_model` | 64 | 32-256 | Attention model dimension |
| `d_inner` | 128 | 64-512 | Attention FFN dimension |
| `n_head` | 4 | 2-8 | Number of attention heads |
| `kernel_size` | 8 | 4-12 | LFM sub-window size |
| `stride` | 4 | 2-8 | LFM stride |
| `dropout_rate` | 0.05 | 0.0-0.3 | Dropout probability |
| `hidden_dims` | (64, 64) | - | Encoder/decoder layers |

### 10.2 Training Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epochs` | 75 | 10-100 | Maximum training epochs |
| `learning_rate` | 5e-4 | 1e-5-1e-3 | Adam learning rate |
| `batch_size` | 64 | 16-256 | Training batch size |
| `patience` | 5 | 3-10 | Early stopping patience |
| `grad_clip` | 2.0 | 0.5-5.0 | Gradient clipping max value |
| `kl_warmup_epochs` | 35 | 5-40 | Epochs for KL annealing |
| `scheduler_t_max` | 10 | 5-20 | CosineAnnealing T_max |

### 10.3 Scorer Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `threshold_method` | "f1_max" | - | "percentile" or "f1_max" |
| `threshold_percentile` | 5.0 | 1-10 | LOW percentile for threshold |
| `hard_criterion_k` | 3 | 1-6 | Min anomalous points to flag (legacy modes) |
| `score_mode` | "single_pass" | - | "single_pass" or "mcmc" |
| `n_samples` | 16 | 8-64 | Latent samples for scoring |
| `decision_mode` | "last_point" | - | Anomaly decision logic |
| `severity_margin` | 0.5 | 0.1-2.0 | Severity decision margin (legacy modes) |
| `outlier_z_threshold` | 3.0 | 2.0-4.0 | Z-score threshold (legacy modes) |

### 10.4 Augmentation Rates

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `point_ano_rate` | 0.05 | 0.0-0.2 | Point anomaly injection rate |
| `seg_ano_rate` | 0.1 | 0.0-0.3 | Segment swap rate |
| `missing_data_rate` | 0.01 | 0.0-0.05 | Missing data injection rate |

---

## File Descriptions

| File | Lines | Description |
|------|-------|-------------|
| `fcvae_model.py` | ~440 | Core FCVAE VAE architecture with frequency conditioning |
| `attention.py` | ~100 | Transformer attention modules |
| `fcvae_scorer.py` | ~580 | NLL-based scoring with multiple threshold methods |
| `fcvae_registry.py` | ~790 | Manages 4 independent combo models |
| `fcvae_streaming_detector.py` | ~490 | Real-time streaming interface |
| `train_fcvae.py` | ~730 | Complete training pipeline with augmentation |
| `fcvae_augment.py` | ~220 | Point/segment/missing data augmentation |
| `evaluate_fcvae.py` | ~1760 | Comprehensive evaluation and plotting |
| `main.py` | ~600 | Dash application for visualization |
