"""
BiLSTM training & inference orchestrator (TensorFlow only) — no UI dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

import config
from backend_setup import backend_label, setup_backend, tf_version
from data_pipeline import SensorDataPipeline
from device import DeviceManager
from model.bilstm import BiLSTMBuilder


@dataclass
class TrainResult:
    """Structured training outcome for the UI (or CLI) to display."""

    model_type: str
    n_samples: int
    n_sequences: int
    metrics_text: str
    history: dict[str, list] = field(default_factory=dict)
    model_path: str = ""
    device: str = ""


class ProgressCallbackFactory:
    """Build a Keras callback that maps epoch progress to 0–100 for the UI."""

    def __init__(
        self,
        total_epochs: int,
        progress_callback: Optional[Callable[[float], None]] = None,
        base: float = 40.0,
        span: float = 50.0,
    ) -> None:
        setup_backend()
        from tensorflow import keras

        self._keras = keras
        self.total_epochs = max(total_epochs, 1)
        self.progress_callback = progress_callback
        self.base = base
        self.span = span

    def build(self) -> Any:
        progress_callback = self.progress_callback
        total_epochs = self.total_epochs
        base = self.base
        span = self.span
        keras = self._keras

        class _CB(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if progress_callback is None:
                    return
                frac = (epoch + 1) / total_epochs
                progress_callback(base + span * frac)

        return _CB()


class ModelTrainer:
    """Train, evaluate, save/load, and run BiLSTM predictions on sensor data."""

    def __init__(self) -> None:
        self.model: Any = None
        self.pipeline: Optional[SensorDataPipeline] = None
        self.model_type: str = config.DEFAULT_MODEL_TYPE
        self.sequence_length: int = config.SEQUENCE_LENGTH
        self.last_device: str = "/CPU:0"
        self.last_history: dict[str, list] = {}

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.pipeline is not None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        rows: Sequence[dict],
        *,
        model_type: str = config.DEFAULT_MODEL_TYPE,
        target_column: str = "label",
        test_size: float = config.DEFAULT_TEST_SPLIT,
        sequence_length: int = config.SEQUENCE_LENGTH,
        lstm_units: int = config.LSTM_UNITS,
        epochs: int = config.DEFAULT_EPOCHS,
        batch_size: int = config.DEFAULT_BATCH_SIZE,
        learning_rate: float = config.DEFAULT_LEARNING_RATE,
        use_gpu: bool = config.DEFAULT_USE_GPU,
        model_dir: str = config.DEFAULT_MODEL_DIR,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> TrainResult:
        """
        Fit a Bidirectional LSTM on sliding windows of sensor readings.

        progress_callback(percent: float) is optional for UI progress bars.
        """
        setup_backend()
        import tensorflow as tf
        from tensorflow import keras

        min_needed = max(config.MIN_TRAIN_SAMPLES, sequence_length + 5)
        if len(rows) < min_needed:
            raise ValueError(
                f"Need at least {min_needed} records to train a BiLSTM "
                f"with sequence_length={sequence_length} (got {len(rows)})."
            )

        def progress(p: float) -> None:
            if progress_callback is not None:
                progress_callback(p)

        progress(5)
        device_info = DeviceManager.configure(use_gpu=use_gpu)
        self.last_device = device_info.device_name
        progress(10)

        self.model_type = model_type
        self.sequence_length = sequence_length
        self.pipeline = SensorDataPipeline(sequence_length=sequence_length)

        batch = self.pipeline.build_training_batch(
            rows, target_column=target_column, fit_scaler=True
        )
        progress(25)

        x_train, x_test, y_train, y_test = SensorDataPipeline.train_test_split(
            batch.x,
            batch.y,
            test_size=test_size,
            random_state=config.RANDOM_STATE,
        )
        progress(35)

        builder = BiLSTMBuilder(
            sequence_length=sequence_length,
            n_features=len(config.FEATURE_COLUMNS),
            lstm_units=lstm_units,
            learning_rate=learning_rate,
            model_type=model_type,
        )

        with tf.device(device_info.device_name):
            self.model = builder.build()
            callbacks = self._make_callbacks(epochs, progress_callback)

            history = self.model.fit(
                x_train,
                y_train,
                validation_split=config.VALIDATION_SPLIT,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks,
            )
        progress(90)

        y_prob = np.asarray(self.model.predict(x_test, verbose=0)).reshape(-1)
        metrics_text = self._format_metrics(y_test, y_prob, model_type)
        metrics_text = (
            f"{builder.summary_text(self.model)}\n\n"
            f"Backend: {backend_label()} {tf_version()}\n"
            f"Device: {device_info.status_text} ({device_info.device_name})\n"
            f"Raw samples: {batch.n_raw_rows} | Sequences: {len(batch.x)}\n"
            f"Train / test sequences: {len(x_train)} / {len(x_test)}\n\n"
            f"{metrics_text}"
        )

        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        self.last_history = hist
        if "loss" in hist:
            metrics_text += f"\nFinal train loss: {hist['loss'][-1]:.6f}"
        if "val_loss" in hist:
            metrics_text += f" | val loss: {hist['val_loss'][-1]:.6f}"
        metrics_text += f"\nEpochs run: {len(hist.get('loss', []))}"

        saved_path = self.save(model_dir)
        metrics_text += f"\n\nModel saved to {saved_path}"
        progress(100)

        return TrainResult(
            model_type=model_type,
            n_samples=batch.n_raw_rows,
            n_sequences=len(batch.x),
            metrics_text=metrics_text,
            history=hist,
            model_path=saved_path,
            device=device_info.device_name,
        )

    def _make_callbacks(
        self,
        epochs: int,
        progress_callback: Optional[Callable[[float], None]],
    ) -> list:
        from tensorflow import keras

        return [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.MIN_LEARNING_RATE,
            ),
            ProgressCallbackFactory(epochs, progress_callback).build(),
        ]

    def _format_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_type: str,
    ) -> str:
        if model_type == "classifier":
            y_pred = (y_prob >= config.CLASSIFICATION_THRESHOLD).astype(int)
            y_true_i = y_true.astype(int)
            acc = float(np.mean(y_pred == y_true_i))
            tp = int(np.sum((y_pred == 1) & (y_true_i == 1)))
            tn = int(np.sum((y_pred == 0) & (y_true_i == 0)))
            fp = int(np.sum((y_pred == 1) & (y_true_i == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true_i == 1)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            return (
                "Training complete!\n\n"
                f"Accuracy : {acc:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall   : {recall:.4f}\n"
                f"F1       : {f1:.4f}\n"
                f"Mean P(anomaly): {float(np.mean(y_prob)):.4f}\n\n"
                f"Confusion Matrix [[TN, FP], [FN, TP]]:\n"
                f"  [[{tn}, {fp}],\n   [{fn}, {tp}]]\n"
            )

        mse = float(np.mean((y_true - y_prob) ** 2))
        mae = float(np.mean(np.abs(y_true - y_prob)))
        rmse = float(np.sqrt(mse))
        return (
            "Training complete!\n\n"
            f"MSE : {mse:.4f}\n"
            f"RMSE: {rmse:.4f}\n"
            f"MAE : {mae:.4f}\n"
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def ensure_loaded(self, model_dir: str = config.DEFAULT_MODEL_DIR) -> None:
        if self.is_ready:
            return
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model available at '{model_dir}'. Train first."
            )
        self.load(str(path))

    def predict_sequence(self, x: np.ndarray) -> dict[str, Any]:
        """Predict from windows of shape (n, seq_len, n_features)."""
        self.ensure_loaded()
        assert self.model is not None

        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        probs = np.asarray(self.model.predict(x, verbose=0)).reshape(-1)

        if self.model_type == "classifier":
            preds = (probs >= config.CLASSIFICATION_THRESHOLD).astype(int)
            return {
                "predictions": preds,
                "probabilities": probs,
                "prediction": int(preds[-1]),
                "confidence": float(
                    probs[-1] if preds[-1] == 1 else 1.0 - probs[-1]
                ),
            }

        return {
            "predictions": probs,
            "prediction": float(probs[-1]),
        }

    def predict_one(
        self,
        values: Sequence[float],
        history_rows: Optional[Sequence[dict]] = None,
    ) -> dict[str, Any]:
        """
        Single-step prediction.

        Prefer temporal context from history_rows (sensor buffer).
        If history is missing, the feature vector is tiled across the window.
        """
        self.ensure_loaded()
        assert self.pipeline is not None

        last = np.asarray(list(values), dtype=np.float64)
        if history_rows:
            history = self.pipeline.extract_features(history_rows)
            if len(history) > 0:
                history = history[:-1]
            x = self.pipeline.sequence_from_vectors(history, last_vector=last)
        else:
            tiled = np.repeat(last.reshape(1, -1), self.sequence_length, axis=0)
            x = self.pipeline.sequence_from_vectors(tiled)

        return self.predict_sequence(x)

    def predict_batch(
        self, rows: Sequence[dict]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sliding-window predictions over a CSV-style row list."""
        self.ensure_loaded()
        assert self.pipeline is not None and self.model is not None

        x = self.pipeline.transform_rows(rows)
        out = self.predict_sequence(x)
        preds = out["predictions"]

        extra: dict[str, Any] = {
            "n_samples": len(rows),
            "n_sequences": len(preds),
        }

        if rows and "label" in rows[0] and self.model_type == "classifier":
            labels = np.array([int(r["label"]) for r in rows], dtype=int)
            y_true = labels[self.sequence_length - 1 :]
            if len(y_true) == len(preds):
                extra["accuracy"] = float(np.mean(y_true == preds.astype(int)))

        return np.asarray(preds), extra

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: str = config.DEFAULT_MODEL_DIR) -> str:
        if not self.is_ready or self.pipeline is None or self.model is None:
            raise RuntimeError("Nothing to save — train a model first.")

        directory = Path(model_dir)
        directory.mkdir(parents=True, exist_ok=True)

        model_path = directory / "model.keras"
        self.model.save(model_path)

        self.pipeline.save_artifacts(
            directory,
            extra_meta={
                "model_type": self.model_type,
                "sequence_length": self.sequence_length,
                "feature_columns": list(config.FEATURE_COLUMNS),
                "backend": "tensorflow",
                "tf_version": tf_version(),
            },
        )
        return str(directory)

    def load(self, model_dir: str = config.DEFAULT_MODEL_DIR) -> None:
        setup_backend()
        from tensorflow import keras

        directory = Path(model_dir)
        self.pipeline, meta = SensorDataPipeline.load_artifacts(directory)
        self.model_type = meta.get("model_type", config.DEFAULT_MODEL_TYPE)
        self.sequence_length = int(
            meta.get("sequence_length", config.SEQUENCE_LENGTH)
        )
        self.model = keras.models.load_model(directory / "model.keras")
