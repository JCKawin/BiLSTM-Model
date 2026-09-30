# Sensor BiLSTM trainer

CustomTkinter app that trains a bidirectional LSTM on multi-sensor time series with TensorFlow.

Keras 3 can run on TensorFlow, JAX, or PyTorch. On Python 3.14, TensorFlow had no installable wheels, so this project briefly fell back to PyTorch. That fallback is gone. If TensorFlow is missing, the Settings tab says how to install it.

TensorFlow needs Python 3.9–3.12. 3.11 is the one to use.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

| File | Role |
| --- | --- |
| `config.py` | Constants |
| `backend_setup.py` | Sets `KERAS_BACKEND=tensorflow` |
| `sensor.py` | Synthetic sensor stream |
| `device.py` | GPU / CPU |
| `data_pipeline.py` | Scaling and sliding windows |
| `model/bilstm.py` | The network |
| `trainer.py` | Train, eval, save, load |
| `gui.py` | The window |
| `main.py` | Entry |

The window can watch a live stream, record or load CSV, train, and predict. A saved run is `models/bilstm/model.keras`, `scaler.npz`, and `meta.json`.
