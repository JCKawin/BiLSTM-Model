"""
Sensor value generation — pure simulation logic, no UI.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any

import numpy as np

import config


class SensorSimulator:
    """Simulates multi-sensor readings with noise, drift, and anomaly labels."""

    def __init__(self) -> None:
        self.base = dict(config.SENSOR_BASE)
        self.time = 0

    def read(self) -> dict[str, Any]:
        """Return one reading from all sensors."""
        self.time += 1
        t = self.time

        temperature = (
            self.base["temperature"]
            + config.TEMP_AMPLITUDE * np.sin(t * config.TEMP_FREQ)
            + random.gauss(0, config.TEMP_NOISE_STD)
        )
        humidity = (
            self.base["humidity"]
            + config.HUMIDITY_AMPLITUDE * np.cos(t * config.HUMIDITY_FREQ)
            + random.gauss(0, config.HUMIDITY_NOISE_STD)
        )
        pressure = self.base["pressure"] + random.gauss(0, config.PRESSURE_NOISE_STD)
        vibration = self.base["vibration"] + abs(
            random.gauss(0, config.VIBRATION_NOISE_STD)
        )
        light = (
            self.base["light"]
            + config.LIGHT_AMPLITUDE * np.sin(t * config.LIGHT_FREQ)
            + random.gauss(0, config.LIGHT_NOISE_STD)
        )

        label = (
            1
            if (
                temperature > config.ANOMALY_TEMP_THRESHOLD
                and vibration > config.ANOMALY_VIBRATION_THRESHOLD
            )
            else 0
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "temperature": round(float(temperature), 2),
            "humidity": round(float(humidity), 2),
            "pressure": round(float(pressure), 2),
            "vibration": round(float(vibration), 3),
            "light": round(float(light), 2),
            "label": label,
        }

    def reset(self) -> None:
        """Reset internal time counter and base values."""
        self.time = 0
        self.base = dict(config.SENSOR_BASE)
