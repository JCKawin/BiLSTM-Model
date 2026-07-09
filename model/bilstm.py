"""
Bidirectional LSTM architecture for multi-sensor time series (TensorFlow / Keras).
"""

from __future__ import annotations

from typing import Any, Optional

import config
from backend_setup import setup_backend


class BiLSTMBuilder:
    """
    Factory for a stacked Bidirectional LSTM network.

    Input shape:  (batch, sequence_length, n_features)
    Output:
      classifier -> sigmoid probability of anomaly (label=1)
      regressor  -> linear scalar target
    """

    def __init__(
        self,
        *,
        sequence_length: int = config.SEQUENCE_LENGTH,
        n_features: int = len(config.FEATURE_COLUMNS),
        lstm_units: int = config.LSTM_UNITS,
        dense_units: int = config.DENSE_UNITS,
        dropout: float = config.DROPOUT_RATE,
        recurrent_dropout: float = config.RECURRENT_DROPOUT,
        merge_mode: str = config.BIDIRECTIONAL_MERGE,
        learning_rate: float = config.DEFAULT_LEARNING_RATE,
        model_type: str = config.DEFAULT_MODEL_TYPE,
    ) -> None:
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.merge_mode = merge_mode
        self.learning_rate = learning_rate
        self.model_type = model_type

    def build(self) -> Any:
        """Construct and compile a tf.keras Model."""
        setup_backend()
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        inputs = keras.Input(
            shape=(self.sequence_length, self.n_features),
            name="sensor_sequence",
        )

        x = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                name="lstm_1",
            ),
            merge_mode=self.merge_mode,
            name="bilstm_1",
        )(inputs)
        x = layers.BatchNormalization(name="bn_1")(x)

        x = layers.Bidirectional(
            layers.LSTM(
                max(self.lstm_units // 2, 8),
                return_sequences=False,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                name="lstm_2",
            ),
            merge_mode=self.merge_mode,
            name="bilstm_2",
        )(x)
        x = layers.BatchNormalization(name="bn_2")(x)

        x = layers.Dense(self.dense_units, activation="relu", name="dense_1")(x)
        x = layers.Dropout(self.dropout, name="dropout_head")(x)

        if self.model_type == "classifier":
            outputs = layers.Dense(1, activation="sigmoid", name="anomaly_prob")(x)
            loss = "binary_crossentropy"
            metrics: list = [
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ]
        else:
            outputs = layers.Dense(1, activation="linear", name="regression_out")(x)
            loss = "mse"
            metrics = ["mae", "mse"]

        model = keras.Model(inputs=inputs, outputs=outputs, name="sensor_bilstm")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics,
        )
        # Touch tf so static analyzers know we depend on it.
        _ = tf.__version__
        return model

    def summary_text(self, model: Optional[Any] = None) -> str:
        """Return architecture hyper-parameters (and optional Keras summary)."""
        from backend_setup import backend_label, tf_version

        lines = [
            "BiLSTM Architecture (TensorFlow)",
            f"  backend         : {backend_label()} {tf_version()}",
            f"  sequence_length : {self.sequence_length}",
            f"  n_features      : {self.n_features}",
            f"  lstm_units      : {self.lstm_units} "
            f"(stack: {self.lstm_units} -> {max(self.lstm_units // 2, 8)})",
            f"  dense_units     : {self.dense_units}",
            f"  dropout         : {self.dropout}",
            f"  recurrent_drop  : {self.recurrent_dropout}",
            f"  merge_mode      : {self.merge_mode}",
            f"  learning_rate   : {self.learning_rate}",
            f"  task            : {self.model_type}",
        ]
        if model is not None:
            buf: list[str] = []
            model.summary(print_fn=buf.append)
            lines.append("")
            lines.extend(buf)
        return "\n".join(lines)


def build_bilstm(**kwargs: Any) -> Any:
    """Convenience wrapper: build a compiled BiLSTM model."""
    return BiLSTMBuilder(**kwargs).build()
