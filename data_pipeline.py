"""
Sensor time-series preprocessing: scaling + sliding-window sequences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

import config


@dataclass
class FeatureScaler:
    """Z-score scaler for multi-feature sensor vectors (numpy-only)."""

    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> FeatureScaler:
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean_=mean.astype(np.float64), scale_=scale.astype(np.float64))

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean_) / self.scale_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        fitted = FeatureScaler.fit(features)
        self.mean_ = fitted.mean_
        self.scale_ = fitted.scale_
        return self.transform(features)

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        return features * self.scale_ + self.mean_

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez(path, mean=self.mean_, scale=self.scale_)

    @classmethod
    def load(cls, path: str | Path) -> FeatureScaler:
        data = np.load(path)
        return cls(mean_=data["mean"], scale_=data["scale"])


@dataclass
class SequenceBatch:
    """Windowed tensors ready for BiLSTM training / inference."""

    x: np.ndarray  # (n, seq_len, n_features)
    y: np.ndarray  # (n,)
    n_raw_rows: int
    sequence_length: int


class SensorDataPipeline:
    """Turn flat sensor records into scaled BiLSTM sequences."""

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] = config.FEATURE_COLUMNS,
        sequence_length: int = config.SEQUENCE_LENGTH,
        scaler: Optional[FeatureScaler] = None,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.sequence_length = sequence_length
        self.scaler = scaler

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_features(self, rows: Sequence[dict]) -> np.ndarray:
        return np.array(
            [[float(r[c]) for c in self.feature_columns] for r in rows],
            dtype=np.float64,
        )

    def extract_target(self, rows: Sequence[dict], target_column: str) -> np.ndarray:
        return np.array([float(r[target_column]) for r in rows], dtype=np.float64)

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    @staticmethod
    def make_sequences(
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sliding windows over time.

        Each X[i] = features[i : i+seq_len]
        Each y[i] = targets[i + seq_len - 1]  (label/value at window end)
        """
        if len(features) < sequence_length:
            raise ValueError(
                f"Need at least {sequence_length} samples for sequences "
                f"(got {len(features)})."
            )

        xs: list[np.ndarray] = []
        ys: list[float] = []
        for i in range(len(features) - sequence_length + 1):
            xs.append(features[i : i + sequence_length])
            ys.append(float(targets[i + sequence_length - 1]))
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def build_training_batch(
        self,
        rows: Sequence[dict],
        *,
        target_column: str = "label",
        fit_scaler: bool = True,
    ) -> SequenceBatch:
        features = self.extract_features(rows)
        targets = self.extract_target(rows, target_column)

        if fit_scaler or self.scaler is None:
            self.scaler = FeatureScaler.fit(features)
        scaled = self.scaler.transform(features)

        x, y = self.make_sequences(scaled, targets, self.sequence_length)
        return SequenceBatch(
            x=x,
            y=y,
            n_raw_rows=len(rows),
            sequence_length=self.sequence_length,
        )

    def transform_rows(self, rows: Sequence[dict]) -> np.ndarray:
        """Scale rows and return windows only (no targets). Shape (n, L, F)."""
        if self.scaler is None:
            raise RuntimeError("Scaler is not fitted. Train or load a model first.")
        features = self.scaler.transform(self.extract_features(rows))
        if len(features) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} rows for inference "
                f"(got {len(features)})."
            )
        x, _ = self.make_sequences(
            features,
            np.zeros(len(features), dtype=np.float64),
            self.sequence_length,
        )
        return x

    def sequence_from_vectors(
        self,
        history: np.ndarray,
        last_vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build a single inference window.

        history: (n, n_features) unscaled features (oldest -> newest)
        last_vector: optional unscaled override for the final timestep
        Returns shape (1, seq_len, n_features) scaled.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler is not fitted. Train or load a model first.")

        hist = np.asarray(history, dtype=np.float64)
        if last_vector is not None:
            last = np.asarray(last_vector, dtype=np.float64).reshape(1, -1)
            hist = np.vstack([hist, last]) if len(hist) else last

        if len(hist) == 0:
            raise ValueError("No history available for sequence construction.")

        if len(hist) < self.sequence_length:
            # Left-pad by repeating the first observation.
            pad = np.repeat(hist[:1], self.sequence_length - len(hist), axis=0)
            hist = np.vstack([pad, hist])
        else:
            hist = hist[-self.sequence_length :]

        scaled = self.scaler.transform(hist)
        return scaled.astype(np.float32)[np.newaxis, ...]

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    @staticmethod
    def train_test_split(
        x: np.ndarray,
        y: np.ndarray,
        test_size: float = config.DEFAULT_TEST_SPLIT,
        random_state: int = config.RANDOM_STATE,
        shuffle: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(x)
        if n < 2:
            raise ValueError("Need at least 2 sequences to split.")

        indices = np.arange(n)
        rng = np.random.default_rng(random_state)
        if shuffle:
            rng.shuffle(indices)

        n_test = max(1, int(round(n * test_size)))
        n_test = min(n_test, n - 1)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

    # ------------------------------------------------------------------
    # Meta I/O
    # ------------------------------------------------------------------

    def save_artifacts(self, directory: str | Path, extra_meta: Optional[dict] = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.scaler is None:
            raise RuntimeError("No scaler to save.")
        self.scaler.save(directory / "scaler.npz")
        meta: dict[str, Any] = {
            "feature_columns": self.feature_columns,
            "sequence_length": self.sequence_length,
        }
        if extra_meta:
            meta.update(extra_meta)
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load_artifacts(cls, directory: str | Path) -> tuple[SensorDataPipeline, dict]:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        scaler = FeatureScaler.load(directory / "scaler.npz")
        pipeline = cls(
            feature_columns=meta.get("feature_columns", config.FEATURE_COLUMNS),
            sequence_length=int(meta.get("sequence_length", config.SEQUENCE_LENGTH)),
            scaler=scaler,
        )
        return pipeline, meta
