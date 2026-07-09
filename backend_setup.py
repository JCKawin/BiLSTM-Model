"""
TensorFlow-only runtime bootstrap.

Must be imported before keras / model / trainer modules.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Literal

BackendName = Literal["tensorflow"]

_BACKEND: BackendName | None = None

_TF_INSTALL_HINT = """
TensorFlow is required (PyTorch is not used in this project).

Install:
  pip install "tensorflow>=2.15"

Python version note:
  TensorFlow currently supports Python 3.9–3.12 (sometimes 3.13).
  You are running Python {py}. If pip cannot find tensorflow wheels,
  create a 3.11 or 3.12 environment, e.g.:

    py -3.11 -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements.txt
""".strip()


def _has_tensorflow() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


def require_tensorflow() -> None:
    """Raise a clear error if TensorFlow is missing."""
    if _has_tensorflow():
        return
    raise ImportError(_TF_INSTALL_HINT.format(py=sys.version.split()[0]))


def setup_backend() -> BackendName:
    """
    Lock Keras to the TensorFlow backend and verify TF is importable.

    Safe to call multiple times.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    # Force TF — never fall back to torch / jax.
    os.environ["KERAS_BACKEND"] = "tensorflow"
    require_tensorflow()

    import tensorflow as tf  # noqa: F401

    _BACKEND = "tensorflow"
    return _BACKEND


def get_backend() -> BackendName:
    if _BACKEND is None:
        return setup_backend()
    return _BACKEND


def backend_label() -> str:
    return "TensorFlow"


def tf_version() -> str:
    setup_backend()
    import tensorflow as tf

    return tf.__version__
