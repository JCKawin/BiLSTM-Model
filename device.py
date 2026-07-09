"""
GPU / CPU device management for TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from backend_setup import setup_backend, tf_version


@dataclass(frozen=True)
class DeviceInfo:
    """Snapshot of available compute devices."""

    use_gpu: bool
    device_name: str
    status_text: str
    gpu_names: tuple[str, ...]
    backend: str = "tensorflow"


class DeviceManager:
    """Configure TensorFlow GPU or CPU placement for training / inference."""

    _memory_growth_done: bool = False

    @classmethod
    def list_gpu_names(cls) -> list[str]:
        setup_backend()
        import tensorflow as tf

        return [gpu.name for gpu in tf.config.list_physical_devices("GPU")]

    @classmethod
    def gpu_available(cls) -> bool:
        return len(cls.list_gpu_names()) > 0

    @classmethod
    def _enable_memory_growth(cls) -> None:
        if cls._memory_growth_done:
            return
        import tensorflow as tf

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except (ValueError, RuntimeError):
                pass
        cls._memory_growth_done = True

    @classmethod
    def configure(cls, use_gpu: bool = config.DEFAULT_USE_GPU) -> DeviceInfo:
        """
        Resolve the active TensorFlow device policy.

        Returns a DeviceInfo with a `tf.device(...)` path like `/GPU:0` or `/CPU:0`.
        """
        setup_backend()
        import tensorflow as tf

        cls._enable_memory_growth()
        gpu_devices = tf.config.list_physical_devices("GPU")
        gpu_names = tuple(d.name for d in gpu_devices)

        if use_gpu and gpu_names:
            status = f"GPU enabled ({len(gpu_names)} device(s))"
            device_name = "/GPU:0"
            active_gpu = True
        elif use_gpu and not gpu_names:
            status = "GPU requested but none detected — using CPU"
            device_name = "/CPU:0"
            active_gpu = False
        else:
            status = "CPU mode"
            device_name = "/CPU:0"
            active_gpu = False

        return DeviceInfo(
            use_gpu=active_gpu,
            device_name=device_name,
            status_text=status,
            gpu_names=gpu_names,
            backend="tensorflow",
        )

    @classmethod
    def resolve_device(cls, use_gpu: bool) -> str:
        return cls.configure(use_gpu=use_gpu).device_name

    @classmethod
    def status_summary(cls, use_gpu: bool) -> str:
        info = cls.configure(use_gpu=use_gpu)
        ver = tf_version()
        if info.gpu_names:
            names = ", ".join(info.gpu_names)
            return f"[TensorFlow {ver}] {info.status_text} | {names}"
        return f"[TensorFlow {ver}] {info.status_text} | no physical GPU"
