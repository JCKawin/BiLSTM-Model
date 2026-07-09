"""
Central configuration — tweak values here without touching app logic.
"""

# ---------------------------------------------------------------------------
# Window / CustomTkinter
# ---------------------------------------------------------------------------
WINDOW_TITLE = "Sensor BiLSTM Trainer — TensorFlow"
WINDOW_GEOMETRY = "1480x920"
WINDOW_MINSIZE = (1100, 700)
UPDATE_INTERVAL_MS = 500  # live sensor refresh rate

APPEARANCE_MODE = "Dark"  # Dark | Light | System
COLOR_THEME = "blue"  # blue | green | dark-blue

# ---------------------------------------------------------------------------
# Theme / colors (matplotlib + accents)
# ---------------------------------------------------------------------------
BG_COLOR = "#1a1a1a"
CARD_BG = "#242424"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLOR = "#3b8ed0"
ANOMALY_COLOR = "#e74c3c"
NORMAL_COLOR = "#2ecc71"
PLOT_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6"]

FONT_FAMILY = "Segoe UI"
MONO_FONT = "Consolas"

# ---------------------------------------------------------------------------
# Data / files
# ---------------------------------------------------------------------------
DEFAULT_CSV_PATH = "sensor_data.csv"
DEFAULT_MODEL_DIR = "models/bilstm"
BUFFER_MAXLEN = 2000
TREE_PREVIEW_ROWS = 200
BULK_GENERATE_DEFAULT = 300
BULK_GENERATE_OPTIONS = [50, 100, 200, 300, 500, 1000]

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "vibration",
    "light",
]
CSV_FIELDNAMES = ["timestamp"] + FEATURE_COLUMNS + ["label"]
TARGET_OPTIONS = ["label", "temperature", "humidity"]

DEFAULT_INPUTS = {
    "temperature": 25.0,
    "humidity": 50.0,
    "pressure": 1013.0,
    "vibration": 0.5,
    "light": 300.0,
}

# ---------------------------------------------------------------------------
# Sensor simulation
# ---------------------------------------------------------------------------
SENSOR_BASE = {
    "temperature": 25.0,
    "humidity": 50.0,
    "pressure": 1013.0,
    "vibration": 0.5,
    "light": 300.0,
}

TEMP_AMPLITUDE = 5.0
TEMP_FREQ = 0.1
TEMP_NOISE_STD = 1.5

HUMIDITY_AMPLITUDE = 10.0
HUMIDITY_FREQ = 0.05
HUMIDITY_NOISE_STD = 3.0

PRESSURE_NOISE_STD = 2.0
VIBRATION_NOISE_STD = 0.3

LIGHT_AMPLITUDE = 200.0
LIGHT_FREQ = 0.2
LIGHT_NOISE_STD = 50.0

ANOMALY_TEMP_THRESHOLD = 28.0
ANOMALY_VIBRATION_THRESHOLD = 0.8

# ---------------------------------------------------------------------------
# BiLSTM model defaults (TensorFlow)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_TYPE = "classifier"  # "classifier" | "regressor"
DEFAULT_TEST_SPLIT = 0.2
DEFAULT_USE_GPU = True

SEQUENCE_LENGTH = 20
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT = 0.0
BIDIRECTIONAL_MERGE = "concat"  # concat | sum | ave | mul

DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
VALIDATION_SPLIT = 0.1

MIN_TRAIN_SAMPLES = 50
RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.5

TEST_SPLIT_RANGE = (0.1, 0.5)
EPOCHS_RANGE = (5, 200)
BATCH_SIZE_OPTIONS = [8, 16, 32, 64, 128, 256]
SEQUENCE_LENGTH_RANGE = (5, 100)
LSTM_UNITS_OPTIONS = [16, 32, 64, 128, 256]
LEARNING_RATE_OPTIONS = ["0.0001", "0.0005", "0.001", "0.005", "0.01"]

# ---------------------------------------------------------------------------
# Live plot
# ---------------------------------------------------------------------------
PLOT_FIGSIZE = (10, 3.6)
PLOT_DPI = 100
PLOT_Y_PADDING = 5
PLOT_MIN_X_RANGE = 100
LOSS_PLOT_FIGSIZE = (6, 2.4)
