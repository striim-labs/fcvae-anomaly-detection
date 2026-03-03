"""
Core retraining pipeline for FCVAE anomaly detection models.

Fine-tunes from current production weights with reduced LR/epochs,
computes oracle threshold from labeled historical data, validates
against the old model, and stages artifacts for hot-swap.
"""

import logging
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add app/ to Python path (same pattern as api/dependencies.py)
_app_dir = str(Path(__file__).resolve().parent.parent / "app")
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from fcvae_augment import AugmentConfig, batch_augment  # noqa: E402
from fcvae_model import FCVAE, FCVAEConfig  # noqa: E402
from fcvae_scorer import FCVAEScorer  # noqa: E402
from transaction_preprocessor import SlidingWindowDataset  # noqa: E402

from api.data_store import WindowDataStore  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class RetrainConfig:
    learning_rate: float = 5e-5  # 10x lower than initial 5e-4
    epochs: int = 20  # vs 30 initial
    kl_warmup_epochs: int = 7  # vs 10 initial
    patience: int = 5
    batch_size: int = 64
    train_fraction: float = 0.8
    max_nll_degradation: float = 0.5
    min_f1_ratio: float = 0.9
    staging_dir: str = "models/fcvae_staging"
    production_dir: str = "models/fcvae"


@dataclass
class RetrainResult:
    combo: str
    status: str  # "success" | "rejected" | "error" | "insufficient_data"
    old_mean_nll: Optional[float] = None
    new_mean_nll: Optional[float] = None
    old_f1: Optional[float] = None
    new_f1: Optional[float] = None
    num_train_windows: int = 0
    num_val_windows: int = 0
    num_anomaly_windows: int = 0
    best_epoch: int = 0
    new_threshold: Optional[float] = None
    old_threshold: Optional[float] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


class RetrainPipeline:

    def __init__(
        self,
        data_store: WindowDataStore,
        config: Optional[RetrainConfig] = None,
        device: str = "cpu",
    ):
        self.data_store = data_store
        self.config = config or RetrainConfig()
        self.device = torch.device(device)
        self.augment_config = AugmentConfig()

    def retrain_combo(
        self,
        combo: str,
        production_dir: Optional[str] = None,
        lookback_days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_windows: int = 50,
    ) -> RetrainResult:
        start_time = time.perf_counter()
        prod_dir = Path(production_dir or self.config.production_dir)

        try:
            return self._retrain_combo_inner(
                combo, prod_dir, lookback_days, start_date, end_date,
                min_windows, start_time,
            )
        except Exception as e:
            logger.exception(f"Retrain failed for {combo}")
            return RetrainResult(
                combo=combo,
                status="error",
                error_message=str(e),
                duration_seconds=time.perf_counter() - start_time,
            )

    def _retrain_combo_inner(
        self,
        combo: str,
        prod_dir: Path,
        lookback_days: Optional[int],
        start_date: Optional[str],
        end_date: Optional[str],
        min_windows: int,
        start_time: float,
    ) -> RetrainResult:
        # --- Step 1: Load normal windows ---
        normal_windows = self.data_store.get_normal_windows(
            combo, start_date=start_date, end_date=end_date,
            lookback_days=lookback_days,
        )
        logger.info(f"[{combo}] Loaded {len(normal_windows)} normal windows")

        # --- Step 2: Sufficiency check ---
        if len(normal_windows) < min_windows:
            return RetrainResult(
                combo=combo,
                status="insufficient_data",
                num_train_windows=len(normal_windows),
                duration_seconds=time.perf_counter() - start_time,
            )

        # --- Step 3: Chronological 80/20 split ---
        split_idx = int(len(normal_windows) * self.config.train_fraction)
        train_windows = normal_windows[:split_idx]
        val_windows = normal_windows[split_idx:]

        train_raw = np.array(
            [w["raw_values"] for w in train_windows], dtype=np.float32,
        )  # (N_train, 24)
        val_raw = np.array(
            [w["raw_values"] for w in val_windows], dtype=np.float32,
        )  # (N_val, 24)

        # --- Step 4: Fit new StandardScaler on training partition only ---
        scaler = StandardScaler()
        scaler.fit(train_raw.reshape(-1, 1))

        # --- Step 5: Normalize train and val ---
        train_norm = (
            scaler.transform(train_raw.reshape(-1, 1))
            .reshape(train_raw.shape)
            .astype(np.float32)
        )
        val_norm = (
            scaler.transform(val_raw.reshape(-1, 1))
            .reshape(val_raw.shape)
            .astype(np.float32)
        )

        # --- Step 6: Create DataLoaders ---
        train_labels = np.zeros_like(train_norm)
        val_labels = np.zeros_like(val_norm)

        train_dataset = SlidingWindowDataset(train_norm, train_labels)
        val_dataset = SlidingWindowDataset(val_norm, val_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False,
        )

        # --- Step 7: Load current production model ---
        combo_dir = prod_dir / combo
        checkpoint = torch.load(
            combo_dir / "model.pt", map_location=self.device, weights_only=False,
        )
        model_config: FCVAEConfig = checkpoint["model_config"]
        old_version = checkpoint.get("model_version", 0)

        model = FCVAE(config=model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

        logger.info(f"[{combo}] Loaded production model v{old_version}")

        # --- Step 8: Fine-tune ---
        history = self._fine_tune(model, train_loader, val_loader, combo)
        model.eval()

        logger.info(
            f"[{combo}] Fine-tuning complete. Best epoch: {history['best_epoch']}"
        )

        # --- Step 9: Oracle threshold from all labeled history ---
        all_windows = self.data_store.get_all_windows(combo)
        all_raw = np.array(
            [w["raw_values"] for w in all_windows], dtype=np.float32,
        )
        all_labels = np.array(
            [w["is_anomaly"] for w in all_windows], dtype=bool,
        )

        # Normalize with new scaler, score with fine-tuned model
        all_norm = (
            scaler.transform(all_raw.reshape(-1, 1))
            .reshape(all_raw.shape)
            .astype(np.float32)
        )
        new_all_scores = self._score_windows(model, all_norm)  # (N, 24)

        normal_mask = ~all_labels
        anomaly_mask = all_labels
        num_anomaly = int(anomaly_mask.sum())

        scorer = FCVAEScorer()

        if anomaly_mask.any():
            # Oracle on last-point scores
            normal_lp = new_all_scores[normal_mask, -1]
            anomaly_lp = new_all_scores[anomaly_mask, -1]
            lp_thresh, _ = FCVAEScorer.find_optimal_threshold(
                normal_lp, anomaly_lp, method="f1_max",
            )
            # Oracle on all-position scores
            normal_flat = new_all_scores[normal_mask].flatten()
            anomaly_flat = new_all_scores[anomaly_mask].flatten()
            pt_thresh, _ = FCVAEScorer.find_optimal_threshold(
                normal_flat, anomaly_flat, method="f1_max",
            )
            scorer.last_point_threshold = lp_thresh
            scorer.point_threshold = pt_thresh
            logger.info(
                f"[{combo}] Oracle thresholds: lp={lp_thresh:.4f}, pt={pt_thresh:.4f}"
            )
        else:
            # Fallback: 5th percentile of normal last-point scores
            normal_lp = new_all_scores[normal_mask, -1]
            fallback = float(np.percentile(normal_lp, 5))
            scorer.last_point_threshold = fallback
            scorer.point_threshold = fallback
            logger.info(
                f"[{combo}] No anomalous windows; fallback threshold={fallback:.4f}"
            )

        scorer.is_fitted = True
        new_threshold = scorer.last_point_threshold

        # --- Step 10: Validation gate ---
        # Reconstruct old model from the checkpoint loaded in step 7
        old_model = FCVAE(config=model_config)
        old_model.load_state_dict(checkpoint["model_state_dict"])
        old_model.to(self.device)
        old_model.eval()

        with open(combo_dir / "scaler.pkl", "rb") as f:
            old_scaler = pickle.load(f)

        old_scorer = FCVAEScorer.load(str(combo_dir / "scorer.pkl"))
        old_threshold = old_scorer.last_point_threshold or old_scorer.point_threshold

        # Score all windows with old model + old scaler normalization
        old_all_norm = (
            old_scaler.transform(all_raw.reshape(-1, 1))
            .reshape(all_raw.shape)
            .astype(np.float32)
        )
        old_all_scores = self._score_windows(old_model, old_all_norm)  # (N, 24)

        # Mean NLL on normal windows (last-point)
        old_mean_nll = float(np.mean(old_all_scores[normal_mask, -1]))
        new_mean_nll = float(np.mean(new_all_scores[normal_mask, -1]))

        # NLL gate: reject if new model reconstructs normal data worse
        # (NLL is negative; more negative = worse reconstruction)
        if new_mean_nll < old_mean_nll - self.config.max_nll_degradation:
            logger.warning(
                f"[{combo}] REJECTED: NLL degradation "
                f"(old={old_mean_nll:.4f}, new={new_mean_nll:.4f}, "
                f"max_deg={self.config.max_nll_degradation})"
            )
            return RetrainResult(
                combo=combo,
                status="rejected",
                old_mean_nll=old_mean_nll,
                new_mean_nll=new_mean_nll,
                num_train_windows=len(train_windows),
                num_val_windows=len(val_windows),
                num_anomaly_windows=num_anomaly,
                best_epoch=history["best_epoch"],
                new_threshold=new_threshold,
                old_threshold=old_threshold,
                error_message="NLL degradation exceeded threshold",
                duration_seconds=time.perf_counter() - start_time,
            )

        # F1 gate: reject if new F1 significantly worse (only when anomalies exist)
        old_f1: Optional[float] = None
        new_f1: Optional[float] = None
        if anomaly_mask.any():
            old_f1 = _compute_f1(old_all_scores[:, -1], old_threshold, all_labels)
            new_f1 = _compute_f1(new_all_scores[:, -1], new_threshold, all_labels)

            if new_f1 < self.config.min_f1_ratio * old_f1:
                logger.warning(
                    f"[{combo}] REJECTED: F1 degradation "
                    f"(old={old_f1:.4f}, new={new_f1:.4f}, "
                    f"min_ratio={self.config.min_f1_ratio})"
                )
                return RetrainResult(
                    combo=combo,
                    status="rejected",
                    old_mean_nll=old_mean_nll,
                    new_mean_nll=new_mean_nll,
                    old_f1=old_f1,
                    new_f1=new_f1,
                    num_train_windows=len(train_windows),
                    num_val_windows=len(val_windows),
                    num_anomaly_windows=num_anomaly,
                    best_epoch=history["best_epoch"],
                    new_threshold=new_threshold,
                    old_threshold=old_threshold,
                    error_message="F1 degradation exceeded threshold",
                    duration_seconds=time.perf_counter() - start_time,
                )

        # --- Step 11: Stage artifacts ---
        staging_dir = Path(self.config.staging_dir) / combo
        staging_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model.config,
                "model_version": old_version + 1,
            },
            staging_dir / "model.pt",
        )

        with open(staging_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        scorer.save(str(staging_dir / "scorer.pkl"))

        with open(staging_dir / "history.pkl", "wb") as f:
            pickle.dump(history, f)

        logger.info(f"[{combo}] Staged artifacts to {staging_dir}")

        return RetrainResult(
            combo=combo,
            status="success",
            old_mean_nll=old_mean_nll,
            new_mean_nll=new_mean_nll,
            old_f1=old_f1,
            new_f1=new_f1,
            num_train_windows=len(train_windows),
            num_val_windows=len(val_windows),
            num_anomaly_windows=num_anomaly,
            best_epoch=history["best_epoch"],
            new_threshold=new_threshold,
            old_threshold=old_threshold,
            duration_seconds=time.perf_counter() - start_time,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fine_tune(
        self,
        model: FCVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        combo: str,
    ) -> Dict[str, Any]:
        """Inline training loop adapted from fcvae_registry.py:185-288."""
        cfg = self.config
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        best_val_loss = float("inf")
        best_model_state: Optional[Dict[str, Any]] = None
        patience_counter = 0

        history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "learning_rates": [],
            "kl_weights": [],
        }

        for epoch in range(cfg.epochs):
            kl_weight = (
                min(1.0, (epoch + 1) / cfg.kl_warmup_epochs)
                if cfg.kl_warmup_epochs > 0
                else 1.0
            )

            # — Training phase —
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                z = batch[2].to(self.device)

                x, y, z = batch_augment(x, y, z, self.augment_config)

                optimizer.zero_grad()
                _mu_x, _var_x, _rec_x, _mu, _var, loss = model(
                    x, mode="train", kl_weight=kl_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= max(num_batches, 1)

            # — Validation phase —
            model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(self.device)
                    _mu_x, _var_x, _rec_x, _mu, _var, loss = model(
                        x, mode="valid",
                    )
                    val_loss += loss.item()
                    num_val_batches += 1

            val_loss /= max(num_val_batches, 1)

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rates"].append(optimizer.param_groups[0]["lr"])
            history["kl_weights"].append(kl_weight)

            # Track best model by validation loss
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                history["best_epoch"] = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == cfg.epochs - 1:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  [{combo}] Epoch {epoch + 1:3d}: "
                    f"train={train_loss:.6f}, val={val_loss:.6f}, "
                    f"lr={lr:.2e}, kl_w={kl_weight:.2f}"
                )

            # Early stopping
            if patience_counter >= cfg.patience:
                logger.info(f"  [{combo}] Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(self.device)

        return history

    def _score_windows(self, model: FCVAE, windows_norm: np.ndarray) -> np.ndarray:
        """Score normalized windows with model.score_single_pass.

        Args:
            model: FCVAE model (should be in eval mode).
            windows_norm: Normalized windows, shape (N, 24).

        Returns:
            NLL scores, shape (N, 24).
        """
        model.eval()
        tensor = (
            torch.FloatTensor(windows_norm).unsqueeze(1).to(self.device)
        )  # (N, 1, 24)
        with torch.no_grad():
            scores = model.score_single_pass(tensor)  # (N, 24)
        return scores.cpu().numpy()


def _compute_f1(
    lp_scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
) -> float:
    """Compute F1 from last-point scores, threshold, and ground-truth labels.

    Uses inverted comparison: score < threshold → predicted anomaly.
    """
    predictions = lp_scores < threshold
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        return float(2.0 * precision * recall / (precision + recall))
    return 0.0
