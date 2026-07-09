"""
Sensor BiLSTM Trainer — CustomTkinter + TensorFlow entry point.

Layout:
  config.py         — tweakable constants
  backend_setup.py  — TensorFlow-only bootstrap
  sensor.py         — SensorSimulator
  device.py         — GPU / CPU (TensorFlow)
  data_pipeline.py  — scaling + sliding windows
  model/bilstm.py   — Bidirectional LSTM (tf.keras)
  trainer.py        — train / predict / save / load
  gui.py            — CustomTkinter UI
  main.py           — launches the app
"""

from __future__ import annotations

import customtkinter as ctk

import config


def main() -> None:
    ctk.set_appearance_mode(config.APPEARANCE_MODE)
    ctk.set_default_color_theme(config.COLOR_THEME)

    # Validate TensorFlow early so the user sees a clear error before the UI.
    try:
        from backend_setup import setup_backend, tf_version

        setup_backend()
        print(f"TensorFlow {tf_version()} ready")
    except ImportError as exc:
        # Still open the UI so Settings can explain install steps.
        print(str(exc))

    from gui import SensorApp

    app = SensorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
