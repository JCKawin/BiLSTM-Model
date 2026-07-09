"""
CustomTkinter GUI — presentation only. Sensor + TF training live elsewhere.
"""

from __future__ import annotations

import csv
import os
import threading
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Deque, Optional

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import config
from device import DeviceManager
from sensor import SensorSimulator
from trainer import ModelTrainer


class SensorApp(ctk.CTk):
    """Main application window built with CustomTkinter."""

    def __init__(self) -> None:
        super().__init__()

        self.title(config.WINDOW_TITLE)
        self.geometry(config.WINDOW_GEOMETRY)
        self.minsize(*config.WINDOW_MINSIZE)

        self.sensor = SensorSimulator()
        self.trainer = ModelTrainer()
        self.data_buffer: Deque[dict] = deque(maxlen=config.BUFFER_MAXLEN)
        self.recording = False
        self.paused = False
        self.filename = config.DEFAULT_CSV_PATH
        self._anomaly_count = 0
        self._total_seen = 0
        self._last_reading: Optional[dict] = None

        # --- state vars ---
        self.model_type = ctk.StringVar(value=config.DEFAULT_MODEL_TYPE)
        self.test_split = ctk.DoubleVar(value=config.DEFAULT_TEST_SPLIT)
        self.target_var = ctk.StringVar(value="label")
        self.epochs_var = ctk.IntVar(value=config.DEFAULT_EPOCHS)
        self.batch_size_var = ctk.StringVar(value=str(config.DEFAULT_BATCH_SIZE))
        self.seq_len_var = ctk.IntVar(value=config.SEQUENCE_LENGTH)
        self.lstm_units_var = ctk.StringVar(value=str(config.LSTM_UNITS))
        self.lr_var = ctk.StringVar(value=str(config.DEFAULT_LEARNING_RATE))
        self.use_gpu_var = ctk.BooleanVar(value=config.DEFAULT_USE_GPU)
        self.device_status_var = ctk.StringVar(value="Device: not configured")
        self.status_var = ctk.StringVar(value="Ready")
        self.stats_var = ctk.StringVar(value="No data loaded")
        self.buffer_count_var = ctk.StringVar(value="Buffer: 0")
        self.anomaly_rate_var = ctk.StringVar(value="Anomalies: 0 (0%)")
        self.live_status_var = ctk.StringVar(value="Live")
        self.appearance_var = ctk.StringVar(value=config.APPEARANCE_MODE)
        self.bulk_count_var = ctk.StringVar(value=str(config.BULK_GENERATE_DEFAULT))
        self.file_path_var = ctk.StringVar(value=self.filename)
        self.progress_var = ctk.DoubleVar(value=0.0)
        self.pred_result_var = ctk.StringVar(value="—")

        self.input_vars: dict[str, ctk.DoubleVar] = {
            f: ctk.DoubleVar(value=config.DEFAULT_INPUTS[f])
            for f in config.FEATURE_COLUMNS
        }
        self.sensor_value_vars: dict[str, ctk.StringVar] = {
            name: ctk.StringVar(value="--")
            for name in list(config.FEATURE_COLUMNS) + ["status"]
        }

        self._build_ui()
        self._refresh_device_status()
        self._schedule_update()

    # ==================================================================
    # Layout
    # ==================================================================

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_tabs()
        self._build_statusbar()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Sensor BiLSTM Trainer",
            font=ctk.CTkFont(family=config.FONT_FAMILY, size=22, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text="TensorFlow · Bidirectional LSTM · Multi-sensor anomaly detection",
            font=ctk.CTkFont(family=config.FONT_FAMILY, size=12),
            text_color=("gray30", "gray70"),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        controls = ctk.CTkFrame(header, fg_color="transparent")
        controls.grid(row=0, column=1, rowspan=2, sticky="e")

        ctk.CTkLabel(controls, text="Theme").pack(side="left", padx=(0, 6))
        ctk.CTkOptionMenu(
            controls,
            values=["Dark", "Light", "System"],
            variable=self.appearance_var,
            command=self._on_appearance_change,
            width=100,
        ).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(
            controls,
            textvariable=self.buffer_count_var,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left", padx=8)
        ctk.CTkLabel(
            controls,
            textvariable=self.anomaly_rate_var,
            font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=8)

    def _build_tabs(self) -> None:
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=16, pady=6)

        self.tab_live = self.tabs.add("Live Monitor")
        self.tab_data = self.tabs.add("Data & Files")
        self.tab_ml = self.tabs.add("Train BiLSTM")
        self.tab_test = self.tabs.add("Test Model")
        self.tab_settings = self.tabs.add("Settings")

        self._build_live_tab()
        self._build_data_tab()
        self._build_ml_tab()
        self._build_test_tab()
        self._build_settings_tab()

    def _build_statusbar(self) -> None:
        bar = ctk.CTkFrame(self, height=32, corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            bar,
            textvariable=self.status_var,
            anchor="w",
            font=ctk.CTkFont(size=12),
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=6)

        ctk.CTkLabel(
            bar,
            textvariable=self.live_status_var,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=1, padx=12)

    # ------------------------------------------------------------------
    # Live Monitor
    # ------------------------------------------------------------------

    def _build_live_tab(self) -> None:
        tab = self.tab_live
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        # Controls
        ctrl = ctk.CTkFrame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        self.record_btn = ctk.CTkButton(
            ctrl,
            text="Start Recording",
            command=self.toggle_recording,
            fg_color="#c0392b",
            hover_color="#e74c3c",
            width=140,
        )
        self.record_btn.pack(side="left", padx=6, pady=8)

        self.pause_btn = ctk.CTkButton(
            ctrl,
            text="Pause Live",
            command=self.toggle_pause,
            width=110,
            fg_color=("gray70", "gray35"),
        )
        self.pause_btn.pack(side="left", padx=6, pady=8)

        ctk.CTkLabel(ctrl, text="CSV path").pack(side="left", padx=(16, 6))
        ctk.CTkEntry(ctrl, textvariable=self.file_path_var, width=280).pack(
            side="left", padx=4
        )
        ctk.CTkButton(ctrl, text="Browse", width=80, command=self.browse_file).pack(
            side="left", padx=6
        )

        # Sensor cards
        cards = ctk.CTkFrame(tab, fg_color="transparent")
        cards.grid(row=1, column=0, sticky="ew", padx=4, pady=8)
        for i in range(6):
            cards.grid_columnconfigure(i, weight=1)

        names = [
            ("temperature", "Temperature"),
            ("humidity", "Humidity"),
            ("pressure", "Pressure"),
            ("vibration", "Vibration"),
            ("light", "Light"),
            ("status", "Status"),
        ]
        self.sensor_cards: dict[str, ctk.CTkFrame] = {}
        self.sensor_value_labels: dict[str, ctk.CTkLabel] = {}

        for i, (key, title) in enumerate(names):
            card = ctk.CTkFrame(cards, corner_radius=12)
            card.grid(row=0, column=i, padx=5, sticky="nsew")
            ctk.CTkLabel(
                card,
                text=title,
                font=ctk.CTkFont(size=12, weight="bold"),
            ).pack(pady=(12, 4))
            val = ctk.CTkLabel(
                card,
                textvariable=self.sensor_value_vars[key],
                font=ctk.CTkFont(size=20, weight="bold"),
            )
            val.pack(pady=(4, 14))
            self.sensor_cards[key] = card
            self.sensor_value_labels[key] = val

        # Plot
        graph = ctk.CTkFrame(tab)
        graph.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        graph.grid_columnconfigure(0, weight=1)
        graph.grid_rowconfigure(0, weight=1)

        self.fig = Figure(
            figsize=config.PLOT_FIGSIZE,
            dpi=config.PLOT_DPI,
            facecolor=config.BG_COLOR,
        )
        self.ax = self.fig.add_subplot(111, facecolor=config.CARD_BG)
        self.ax.set_title("Live Sensor Stream", color="white", fontsize=11)
        self.ax.set_xlabel("Sample", color="white")
        self.ax.set_ylabel("Value", color="white")
        self.ax.tick_params(colors="white")
        self.ax.grid(True, alpha=0.25, color="gray")

        self.lines: dict = {}
        for i, key in enumerate(config.FEATURE_COLUMNS):
            (line,) = self.ax.plot(
                [],
                [],
                color=config.PLOT_COLORS[i % len(config.PLOT_COLORS)],
                label=key.title(),
                linewidth=1.5,
            )
            self.lines[key] = line
        self.ax.legend(
            loc="upper left",
            facecolor=config.CARD_BG,
            edgecolor="gray",
            labelcolor="white",
            fontsize=8,
        )
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Data tab
    # ------------------------------------------------------------------

    def _build_data_tab(self) -> None:
        tab = self.tab_data
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ops = ctk.CTkFrame(tab)
        ops.grid(row=0, column=0, sticky="ew", padx=4, pady=8)

        buttons = [
            ("Load CSV", self.load_csv),
            ("Save Buffer", self.save_buffer),
            ("Clear Buffer", self.clear_buffer),
            ("Show Statistics", self.show_stats),
            ("Refresh Table", self.refresh_tree),
            ("Export Stats", self.export_stats),
        ]
        for text, cmd in buttons:
            ctk.CTkButton(ops, text=text, command=cmd, width=120).pack(
                side="left", padx=5, pady=8
            )

        gen = ctk.CTkFrame(ops, fg_color="transparent")
        gen.pack(side="right", padx=8)
        ctk.CTkLabel(gen, text="Bulk generate").pack(side="left", padx=4)
        ctk.CTkOptionMenu(
            gen,
            values=[str(x) for x in config.BULK_GENERATE_OPTIONS],
            variable=self.bulk_count_var,
            width=90,
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            gen,
            text="Generate Samples",
            command=self.bulk_generate,
            fg_color="#16a085",
            hover_color="#1abc9c",
            width=140,
        ).pack(side="left", padx=4)

        # Treeview host (ttk — CTk has no table widget)
        preview = ctk.CTkFrame(tab)
        preview.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        preview.grid_columnconfigure(0, weight=1)
        preview.grid_rowconfigure(0, weight=1)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Data.Treeview",
            background="#2b2b2b",
            foreground="white",
            fieldbackground="#2b2b2b",
            rowheight=24,
        )
        style.configure(
            "Data.Treeview.Heading",
            background="#1f538d",
            foreground="white",
            font=(config.FONT_FAMILY, 10, "bold"),
        )
        style.map("Data.Treeview", background=[("selected", "#1f538d")])

        tree_host = tk.Frame(preview, bg="#2b2b2b")
        tree_host.pack(fill="both", expand=True, padx=6, pady=6)

        self.tree = ttk.Treeview(
            tree_host,
            columns=config.CSV_FIELDNAMES,
            show="headings",
            height=18,
            style="Data.Treeview",
        )
        for col in config.CSV_FIELDNAMES:
            self.tree.heading(col, text=col.title())
            self.tree.column(col, width=140, anchor="center")

        yscroll = ttk.Scrollbar(tree_host, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        ctk.CTkLabel(
            tab,
            textvariable=self.stats_var,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(family=config.MONO_FONT, size=12),
        ).grid(row=2, column=0, sticky="ew", padx=8, pady=8)

    # ------------------------------------------------------------------
    # Train tab
    # ------------------------------------------------------------------

    def _build_ml_tab(self) -> None:
        tab = self.tab_ml
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1)

        # Device card
        device = ctk.CTkFrame(tab)
        device.grid(row=0, column=0, sticky="ew", padx=4, pady=(8, 4))

        ctk.CTkLabel(
            device,
            text="Compute Device (TensorFlow)",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(10, 4))

        row = ctk.CTkFrame(device, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 6))

        ctk.CTkSwitch(
            row,
            text="Use GPU (CUDA)",
            variable=self.use_gpu_var,
            command=self._on_gpu_toggle,
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            row,
            text="Refresh Device",
            width=120,
            command=self._refresh_device_status,
        ).pack(side="left", padx=10)
        ctk.CTkButton(
            row,
            text="Load Model",
            width=110,
            command=self.load_model_dir,
        ).pack(side="left", padx=6)

        ctk.CTkLabel(
            device,
            textvariable=self.device_status_var,
            anchor="w",
            font=ctk.CTkFont(size=12),
        ).pack(fill="x", padx=14, pady=(0, 12))

        # Hyperparams
        cfg = ctk.CTkFrame(tab)
        cfg.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        cfg.grid_columnconfigure((1, 3, 5), weight=1)

        ctk.CTkLabel(
            cfg,
            text="BiLSTM Hyperparameters",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=6, sticky="w", padx=12, pady=(10, 8))

        ctk.CTkLabel(cfg, text="Task").grid(row=1, column=0, sticky="w", padx=12, pady=6)
        ctk.CTkSegmentedButton(
            cfg,
            values=["classifier", "regressor"],
            variable=self.model_type,
        ).grid(row=1, column=1, sticky="ew", padx=6, pady=6)

        ctk.CTkLabel(cfg, text="Target").grid(
            row=1, column=2, sticky="w", padx=12, pady=6
        )
        ctk.CTkOptionMenu(
            cfg, values=config.TARGET_OPTIONS, variable=self.target_var
        ).grid(row=1, column=3, sticky="ew", padx=6, pady=6)

        ctk.CTkLabel(cfg, text="Test Split").grid(
            row=1, column=4, sticky="w", padx=12, pady=6
        )
        split_frame = ctk.CTkFrame(cfg, fg_color="transparent")
        split_frame.grid(row=1, column=5, sticky="ew", padx=6, pady=6)
        self.split_label = ctk.CTkLabel(split_frame, text=f"{config.DEFAULT_TEST_SPLIT:.2f}")
        self.split_label.pack(side="right")
        ctk.CTkSlider(
            split_frame,
            from_=config.TEST_SPLIT_RANGE[0],
            to=config.TEST_SPLIT_RANGE[1],
            number_of_steps=8,
            variable=self.test_split,
            command=self._on_split_slide,
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkLabel(cfg, text="Seq Length").grid(
            row=2, column=0, sticky="w", padx=12, pady=6
        )
        seq_frame = ctk.CTkFrame(cfg, fg_color="transparent")
        seq_frame.grid(row=2, column=1, sticky="ew", padx=6, pady=6)
        self.seq_label = ctk.CTkLabel(seq_frame, text=str(config.SEQUENCE_LENGTH))
        self.seq_label.pack(side="right")
        ctk.CTkSlider(
            seq_frame,
            from_=config.SEQUENCE_LENGTH_RANGE[0],
            to=config.SEQUENCE_LENGTH_RANGE[1],
            number_of_steps=95,
            variable=self.seq_len_var,
            command=self._on_seq_slide,
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkLabel(cfg, text="LSTM Units").grid(
            row=2, column=2, sticky="w", padx=12, pady=6
        )
        ctk.CTkOptionMenu(
            cfg,
            values=[str(x) for x in config.LSTM_UNITS_OPTIONS],
            variable=self.lstm_units_var,
        ).grid(row=2, column=3, sticky="ew", padx=6, pady=6)

        ctk.CTkLabel(cfg, text="Epochs").grid(
            row=2, column=4, sticky="w", padx=12, pady=6
        )
        ep_frame = ctk.CTkFrame(cfg, fg_color="transparent")
        ep_frame.grid(row=2, column=5, sticky="ew", padx=6, pady=6)
        self.ep_label = ctk.CTkLabel(ep_frame, text=str(config.DEFAULT_EPOCHS))
        self.ep_label.pack(side="right")
        ctk.CTkSlider(
            ep_frame,
            from_=config.EPOCHS_RANGE[0],
            to=config.EPOCHS_RANGE[1],
            number_of_steps=39,
            variable=self.epochs_var,
            command=self._on_epoch_slide,
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkLabel(cfg, text="Batch Size").grid(
            row=3, column=0, sticky="w", padx=12, pady=6
        )
        ctk.CTkOptionMenu(
            cfg,
            values=[str(x) for x in config.BATCH_SIZE_OPTIONS],
            variable=self.batch_size_var,
        ).grid(row=3, column=1, sticky="ew", padx=6, pady=6)

        ctk.CTkLabel(cfg, text="Learning Rate").grid(
            row=3, column=2, sticky="w", padx=12, pady=6
        )
        ctk.CTkOptionMenu(
            cfg,
            values=config.LEARNING_RATE_OPTIONS,
            variable=self.lr_var,
        ).grid(row=3, column=3, sticky="ew", padx=6, pady=(6, 12))

        # Train controls
        actions = ctk.CTkFrame(tab, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=4, pady=8)

        self.train_btn = ctk.CTkButton(
            actions,
            text="Train BiLSTM",
            command=self.train_model,
            height=40,
            width=180,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#27ae60",
            hover_color="#2ecc71",
        )
        self.train_btn.pack(side="left", padx=8)

        ctk.CTkButton(
            actions,
            text="Clear Log",
            width=100,
            command=lambda: self.train_result.delete("1.0", "end"),
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            actions,
            text="Save Metrics",
            width=110,
            command=self.save_train_metrics,
        ).pack(side="left", padx=6)

        self.progress = ctk.CTkProgressBar(actions, width=280)
        self.progress.pack(side="right", padx=12)
        self.progress.set(0)

        # Results + loss plot
        bottom = ctk.CTkFrame(tab, fg_color="transparent")
        bottom.grid(row=3, column=0, sticky="nsew", padx=4, pady=4)
        bottom.grid_columnconfigure(0, weight=3)
        bottom.grid_columnconfigure(1, weight=2)
        bottom.grid_rowconfigure(0, weight=1)

        self.train_result = ctk.CTkTextbox(
            bottom,
            font=ctk.CTkFont(family=config.MONO_FONT, size=12),
        )
        self.train_result.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        loss_frame = ctk.CTkFrame(bottom)
        loss_frame.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(
            loss_frame,
            text="Training Curves",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 0))

        self.loss_fig = Figure(
            figsize=config.LOSS_PLOT_FIGSIZE,
            dpi=100,
            facecolor=config.BG_COLOR,
        )
        self.loss_ax = self.loss_fig.add_subplot(111, facecolor=config.CARD_BG)
        self.loss_ax.set_title("Loss", color="white", fontsize=10)
        self.loss_ax.tick_params(colors="white", labelsize=8)
        self.loss_ax.grid(True, alpha=0.25)
        self.loss_fig.tight_layout()
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, master=loss_frame)
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # ------------------------------------------------------------------
    # Test tab
    # ------------------------------------------------------------------

    def _build_test_tab(self) -> None:
        tab = self.tab_test
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        manual = ctk.CTkFrame(tab)
        manual.grid(row=0, column=0, sticky="ew", padx=4, pady=8)
        manual.grid_columnconfigure((1, 3, 5), weight=1)

        ctk.CTkLabel(
            manual,
            text="Manual Prediction",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=6, sticky="w", padx=12, pady=(10, 8))

        for i, field in enumerate(config.FEATURE_COLUMNS):
            r, c = divmod(i, 3)
            ctk.CTkLabel(manual, text=field.title()).grid(
                row=r + 1, column=c * 2, sticky="w", padx=12, pady=6
            )
            ctk.CTkEntry(manual, textvariable=self.input_vars[field]).grid(
                row=r + 1, column=c * 2 + 1, sticky="ew", padx=6, pady=6
            )

        btn_row = ctk.CTkFrame(manual, fg_color="transparent")
        btn_row.grid(row=3, column=0, columnspan=6, sticky="ew", padx=8, pady=10)

        ctk.CTkButton(
            btn_row,
            text="Predict",
            command=self.manual_predict,
            width=120,
            fg_color="#2980b9",
            hover_color="#3498db",
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            btn_row,
            text="Fill From Live",
            command=self.fill_from_live,
            width=130,
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            btn_row,
            text="Live Predict",
            command=self.live_predict,
            width=120,
        ).pack(side="left", padx=6)
        ctk.CTkLabel(
            btn_row,
            textvariable=self.pred_result_var,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side="right", padx=12)

        batch = ctk.CTkFrame(tab)
        batch.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ctk.CTkLabel(
            batch,
            text="Batch / Model",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(10, 4))
        brow = ctk.CTkFrame(batch, fg_color="transparent")
        brow.pack(fill="x", padx=8, pady=(0, 10))
        ctk.CTkButton(
            brow, text="Batch Test CSV", command=self.batch_test, width=140
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            brow, text="Load Model", command=self.load_model_dir, width=120
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            brow,
            text="Clear Log",
            command=lambda: self.test_result.delete("1.0", "end"),
            width=100,
        ).pack(side="left", padx=6)

        self.test_result = ctk.CTkTextbox(
            tab,
            font=ctk.CTkFont(family=config.MONO_FONT, size=12),
        )
        self.test_result.grid(row=2, column=0, sticky="nsew", padx=4, pady=8)

    # ------------------------------------------------------------------
    # Settings tab
    # ------------------------------------------------------------------

    def _build_settings_tab(self) -> None:
        tab = self.tab_settings
        box = ctk.CTkScrollableFrame(tab)
        box.pack(fill="both", expand=True, padx=8, pady=8)

        ctk.CTkLabel(
            box,
            text="Application Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", pady=(4, 12))

        ctk.CTkLabel(
            box,
            text=(
                "Why you previously saw PyTorch\n"
                "————————————————————————————\n"
                "This project is TensorFlow-only now.\n\n"
                "Earlier, Keras 3 was allowed to fall back to a PyTorch backend\n"
                "because TensorFlow has no official wheels for Python 3.14 yet.\n"
                "That fallback is removed. The stack is pure TensorFlow / tf.keras.\n\n"
                "If import fails, install TF on Python 3.11 or 3.12:\n"
                "  py -3.11 -m venv .venv\n"
                "  .venv\\Scripts\\activate\n"
                "  pip install -r requirements.txt\n"
            ),
            justify="left",
            font=ctk.CTkFont(family=config.MONO_FONT, size=12),
        ).pack(anchor="w", pady=8)

        ctk.CTkLabel(
            box,
            text="Paths & defaults are editable in config.py",
            font=ctk.CTkFont(size=12),
            text_color=("gray30", "gray70"),
        ).pack(anchor="w", pady=8)

        ctk.CTkButton(
            box,
            text="Open models folder",
            command=self.open_models_folder,
            width=180,
        ).pack(anchor="w", pady=6)

    # ==================================================================
    # Slider labels
    # ==================================================================

    def _on_split_slide(self, value: float) -> None:
        self.split_label.configure(text=f"{float(value):.2f}")

    def _on_seq_slide(self, value: float) -> None:
        self.seq_label.configure(text=str(int(round(float(value)))))

    def _on_epoch_slide(self, value: float) -> None:
        self.ep_label.configure(text=str(int(round(float(value)))))

    def _on_appearance_change(self, mode: str) -> None:
        ctk.set_appearance_mode(mode)

    # ==================================================================
    # Device
    # ==================================================================

    def _on_gpu_toggle(self) -> None:
        self._refresh_device_status()

    def _refresh_device_status(self) -> None:
        try:
            summary = DeviceManager.status_summary(bool(self.use_gpu_var.get()))
            self.device_status_var.set(summary)
        except ImportError as exc:
            self.device_status_var.set(f"TensorFlow missing — see Settings tab")
            self.status_var.set(str(exc).split("\n")[0])
        except Exception as exc:
            self.device_status_var.set(f"Device error: {exc}")

    # ==================================================================
    # Live loop
    # ==================================================================

    def _schedule_update(self) -> None:
        if not self.paused:
            self._update_sensors()
        self.after(config.UPDATE_INTERVAL_MS, self._schedule_update)

    def _update_sensors(self) -> None:
        reading = self.sensor.read()
        self._last_reading = reading
        self.data_buffer.append(reading)
        self._total_seen += 1
        if reading.get("label", 0) == 1:
            self._anomaly_count += 1

        self.buffer_count_var.set(f"Buffer: {len(self.data_buffer)}")
        rate = (
            100.0 * self._anomaly_count / self._total_seen if self._total_seen else 0.0
        )
        self.anomaly_rate_var.set(
            f"Anomalies: {self._anomaly_count} ({rate:.1f}%)"
        )

        self._refresh_sensor_cards(reading)
        self._refresh_plot()
        if self.recording:
            self._append_csv_row(reading)

    def _refresh_sensor_cards(self, reading: dict) -> None:
        for key in config.FEATURE_COLUMNS:
            self.sensor_value_vars[key].set(str(reading[key]))

        is_anomaly = reading.get("label", 0) == 1
        self.sensor_value_vars["status"].set("ANOMALY" if is_anomaly else "NORMAL")
        color = config.ANOMALY_COLOR if is_anomaly else config.NORMAL_COLOR
        self.sensor_value_labels["status"].configure(text_color=color)

    def _refresh_plot(self) -> None:
        if len(self.data_buffer) <= 1:
            return
        x = list(range(len(self.data_buffer)))
        for key, line in self.lines.items():
            y = [d[key] for d in self.data_buffer]
            line.set_data(x, y)
        self.ax.set_xlim(0, max(config.PLOT_MIN_X_RANGE, len(self.data_buffer)))
        all_vals = [d[k] for d in self.data_buffer for k in self.lines]
        pad = config.PLOT_Y_PADDING
        self.ax.set_ylim(min(all_vals) - pad, max(all_vals) + pad)
        self.canvas.draw_idle()

    def _update_loss_plot(self, history: dict[str, list]) -> None:
        self.loss_ax.clear()
        self.loss_ax.set_facecolor(config.CARD_BG)
        self.loss_ax.tick_params(colors="white", labelsize=8)
        self.loss_ax.grid(True, alpha=0.25)
        if "loss" in history:
            self.loss_ax.plot(history["loss"], color="#3498db", label="train loss")
        if "val_loss" in history:
            self.loss_ax.plot(history["val_loss"], color="#e74c3c", label="val loss")
        if "accuracy" in history:
            ax2 = self.loss_ax.twinx()
            ax2.plot(history["accuracy"], color="#2ecc71", alpha=0.7, label="acc")
            ax2.tick_params(colors="white", labelsize=8)
        self.loss_ax.set_title("Training Curves", color="white", fontsize=10)
        self.loss_ax.legend(loc="upper right", fontsize=8, labelcolor="white")
        self.loss_fig.tight_layout()
        self.loss_canvas.draw_idle()

    # ==================================================================
    # Recording / files
    # ==================================================================

    def toggle_recording(self) -> None:
        if not self.recording:
            self.recording = True
            self.record_btn.configure(text="Stop Recording", fg_color="#7f8c8d")
            self.status_var.set("Recording...")
            self.filename = self.file_path_var.get()
            if not os.path.exists(self.filename):
                with open(self.filename, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=config.CSV_FIELDNAMES).writeheader()
        else:
            self.recording = False
            self.record_btn.configure(text="Start Recording", fg_color="#c0392b")
            self.status_var.set("Recording stopped")

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.configure(text="Resume Live")
            self.live_status_var.set("Paused")
            self.status_var.set("Live updates paused")
        else:
            self.pause_btn.configure(text="Pause Live")
            self.live_status_var.set("Live")
            self.status_var.set("Live updates resumed")

    def browse_file(self) -> None:
        fname = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if fname:
            self.file_path_var.set(fname)

    def _append_csv_row(self, reading: dict) -> None:
        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=config.CSV_FIELDNAMES).writerow(reading)

    def bulk_generate(self) -> None:
        n = int(self.bulk_count_var.get())
        for _ in range(n):
            reading = self.sensor.read()
            self.data_buffer.append(reading)
            self._total_seen += 1
            if reading.get("label", 0) == 1:
                self._anomaly_count += 1
        self.buffer_count_var.set(f"Buffer: {len(self.data_buffer)}")
        self.refresh_tree()
        self.status_var.set(f"Generated {n} synthetic samples into buffer")

    def load_csv(self) -> None:
        fname = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not fname:
            return
        try:
            with open(fname, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.data_buffer.clear()
            self._anomaly_count = 0
            for row in rows:
                item = {
                    "timestamp": row["timestamp"],
                    "temperature": float(row["temperature"]),
                    "humidity": float(row["humidity"]),
                    "pressure": float(row["pressure"]),
                    "vibration": float(row["vibration"]),
                    "light": float(row["light"]),
                    "label": int(row["label"]),
                }
                self.data_buffer.append(item)
                if item["label"] == 1:
                    self._anomaly_count += 1
            self._total_seen = len(self.data_buffer)
            self.filename = fname
            self.file_path_var.set(fname)
            self.refresh_tree()
            self.buffer_count_var.set(f"Buffer: {len(self.data_buffer)}")
            self.status_var.set(f"Loaded {len(self.data_buffer)} records from {fname}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load CSV: {exc}")

    def refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        rows = list(self.data_buffer)
        # Show the newest rows up to TREE_PREVIEW_ROWS
        preview = rows[-config.TREE_PREVIEW_ROWS :]
        for row in preview:
            self.tree.insert(
                "",
                "end",
                values=tuple(row[c] for c in config.CSV_FIELDNAMES),
            )

    def save_buffer(self) -> None:
        if not self.data_buffer:
            messagebox.showwarning("Warning", "No data to save!")
            return
        fname = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not fname:
            return
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=config.CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.data_buffer)
        self.status_var.set(f"Saved {len(self.data_buffer)} records to {fname}")

    def clear_buffer(self) -> None:
        self.data_buffer.clear()
        self._anomaly_count = 0
        self._total_seen = 0
        self.refresh_tree()
        self.buffer_count_var.set("Buffer: 0")
        self.anomaly_rate_var.set("Anomalies: 0 (0%)")
        self.stats_var.set("Buffer cleared")
        self.status_var.set("Buffer cleared")

    def show_stats(self) -> None:
        if not self.data_buffer:
            messagebox.showwarning("Warning", "No data available!")
            return
        data = list(self.data_buffer)
        lines = []
        for key in config.FEATURE_COLUMNS:
            vals = [d[key] for d in data]
            lines.append(
                f"{key.title()}: mean={np.mean(vals):.2f}  std={np.std(vals):.2f}  "
                f"min={min(vals):.2f}  max={max(vals):.2f}"
            )
        anomalies = sum(1 for d in data if d["label"] == 1)
        lines.append(f"\nTotal records: {len(data)}")
        lines.append(f"Anomalies: {anomalies} ({100 * anomalies / len(data):.1f}%)")
        self.stats_var.set("\n".join(lines))

    def export_stats(self) -> None:
        if not self.data_buffer:
            messagebox.showwarning("Warning", "No data available!")
            return
        self.show_stats()
        fname = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if not fname:
            return
        Path(fname).write_text(self.stats_var.get(), encoding="utf-8")
        self.status_var.set(f"Stats exported to {fname}")

    def open_models_folder(self) -> None:
        path = Path(config.DEFAULT_MODEL_DIR)
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path.resolve()))  # type: ignore[attr-defined]
        except Exception:
            self.status_var.set(str(path.resolve()))

    # ==================================================================
    # Training
    # ==================================================================

    def train_model(self) -> None:
        seq_len = int(round(float(self.seq_len_var.get())))
        min_needed = max(config.MIN_TRAIN_SAMPLES, seq_len + 5)
        if len(self.data_buffer) < min_needed:
            messagebox.showwarning(
                "Warning",
                f"Need at least {min_needed} records for seq_length={seq_len} "
                f"(buffer has {len(self.data_buffer)}).\n"
                "Use Live recording, Load CSV, or Bulk Generate.",
            )
            return
        self._refresh_device_status()
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self) -> None:
        self.after(0, lambda: self.train_btn.configure(state="disabled"))
        self.after(0, lambda: self.progress.set(0))
        try:
            rows = list(self.data_buffer)

            def on_progress(percent: float) -> None:
                self.after(0, lambda p=percent: self.progress.set(min(p, 100) / 100.0))

            result = self.trainer.train(
                rows,
                model_type=self.model_type.get(),
                target_column=self.target_var.get(),
                test_size=float(self.test_split.get()),
                sequence_length=int(round(float(self.seq_len_var.get()))),
                lstm_units=int(self.lstm_units_var.get()),
                epochs=int(round(float(self.epochs_var.get()))),
                batch_size=int(self.batch_size_var.get()),
                learning_rate=float(self.lr_var.get()),
                use_gpu=bool(self.use_gpu_var.get()),
                model_dir=config.DEFAULT_MODEL_DIR,
                progress_callback=on_progress,
            )

            def show_ok() -> None:
                self.train_result.delete("1.0", "end")
                self.train_result.insert("end", result.metrics_text)
                self.progress.set(1.0)
                self.status_var.set(
                    f"BiLSTM trained on {result.device} "
                    f"({result.n_sequences} sequences)"
                )
                self._update_loss_plot(result.history)
                self._refresh_device_status()

            self.after(0, show_ok)
        except Exception as exc:
            self.after(0, lambda e=exc: messagebox.showerror("Training Error", str(e)))
        finally:
            self.after(0, lambda: self.train_btn.configure(state="normal"))

    def save_train_metrics(self) -> None:
        text = self.train_result.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Warning", "No training metrics to save.")
            return
        fname = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if fname:
            Path(fname).write_text(text, encoding="utf-8")
            self.status_var.set(f"Metrics saved to {fname}")

    def load_model_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=config.DEFAULT_MODEL_DIR)
        if not path:
            # try default
            path = config.DEFAULT_MODEL_DIR
        try:
            self.trainer.load(path)
            self.status_var.set(f"Loaded model from {path}")
            messagebox.showinfo("Model", f"Loaded BiLSTM from:\n{path}")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    # ==================================================================
    # Testing
    # ==================================================================

    def fill_from_live(self) -> None:
        if not self._last_reading:
            messagebox.showwarning("Warning", "No live reading yet.")
            return
        for f in config.FEATURE_COLUMNS:
            self.input_vars[f].set(float(self._last_reading[f]))
        self.status_var.set("Filled inputs from latest live sample")

    def live_predict(self) -> None:
        self.fill_from_live()
        self.manual_predict()

    def manual_predict(self) -> None:
        try:
            self.trainer.ensure_loaded()
        except FileNotFoundError as exc:
            messagebox.showwarning("Warning", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        try:
            values = [float(self.input_vars[f].get()) for f in config.FEATURE_COLUMNS]
            history = list(self.data_buffer)
            out = self.trainer.predict_one(values, history_rows=history or None)
            pred = out["prediction"]
            if "confidence" in out:
                label = "ANOMALY" if int(pred) == 1 else "NORMAL"
                text = (
                    f"Prediction: {label} ({pred})\n"
                    f"Confidence: {out['confidence']:.4f}\n"
                )
                if "probabilities" in out:
                    text += f"P(anomaly): {float(out['probabilities'][-1]):.4f}\n"
                self.pred_result_var.set(label)
            else:
                text = f"Prediction: {float(pred):.4f}\n"
                self.pred_result_var.set(f"{float(pred):.3f}")
            self.test_result.insert("end", text + "\n")
            self.test_result.see("end")
        except Exception as exc:
            messagebox.showerror("Prediction Error", str(exc))

    def batch_test(self) -> None:
        fname = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not fname:
            return
        try:
            self.trainer.ensure_loaded()
        except Exception as exc:
            messagebox.showwarning("Warning", str(exc))
            return

        try:
            with open(fname, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            preds, extra = self.trainer.predict_batch(rows)
            text = (
                f"Batch Test ({extra['n_samples']} rows → "
                f"{extra.get('n_sequences', len(preds))} windows)\n"
            )
            text += f"Predictions (first 50): {list(preds[:50])}\n"
            if "accuracy" in extra:
                text += f"Accuracy vs Actual: {extra['accuracy']:.4f}\n"
            self.test_result.insert("end", text + "\n")
            self.test_result.see("end")
        except Exception as exc:
            messagebox.showerror("Batch Test Error", str(exc))
