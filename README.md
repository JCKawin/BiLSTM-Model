# Sensor BiLSTM Trainer (TensorFlow + CustomTkinter)

Professional **Bidirectional LSTM** trainer for multi-sensor time series.

- **UI:** CustomTkinter (modern dark theme, tabs, switches, sliders)
- **ML:** **TensorFlow / tf.keras only** (no PyTorch)

## Why PyTorch showed up before

Keras 3 can run on TensorFlow, JAX, or PyTorch. On **Python 3.14**, TensorFlow
had no installable wheels, so the project temporarily fell back to a
**PyTorch backend** so training still worked.

That fallback is **removed**. The app is TensorFlow-only. If TF is missing,
you get a clear install error (see Settings tab in the app).

### Recommended environment

```bash
# TensorFlow needs Python 3.9–3.12 (3.11 recommended)
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Project layout

| Module | Role |
|--------|------|
| `config.py` | All tweakable constants |
| `backend_setup.py` | Forces `KERAS_BACKEND=tensorflow` |
| `sensor.py` | Synthetic sensor stream |
| `device.py` | TF GPU / CPU toggle |
| `data_pipeline.py` | Scaling + sliding windows |
| `model/bilstm.py` | BiLSTM (`tf.keras`) |
| `trainer.py` | Train / eval / save / load |
| `gui.py` | CustomTkinter UI |
| `main.py` | Entry point |

## Features

- Live sensor monitor + matplotlib stream
- Record to CSV / load CSV / bulk-generate samples
- GPU switch for TensorFlow CUDA
- BiLSTM hyperparams (seq length, units, epochs, LR, …)
- Training metrics log + live loss curves
- Manual / live / batch prediction
- Dark / Light / System appearance

## Artifacts

`models/bilstm/` → `model.keras`, `scaler.npz`, `meta.json`
