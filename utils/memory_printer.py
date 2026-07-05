import logging

import psutil
import torch

logger = logging.getLogger(__name__)


def print_memory_usage(label: str = "") -> None:
    """Log current GPU and CPU memory usage at any point in the pipeline.

    Args:
        label: Optional tag to identify where in the pipeline you are.
    """
    prefix = f"[{label}] " if label else ""

    # GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        logger.info(
            "%sGPU  — allocated: %.2fGB | reserved: %.2fGB | free: %.2fGB | total: %.2fGB",
            prefix, allocated, reserved, free, total,
        )
    else:
        logger.info("%sGPU  — not available", prefix)

    # CPU / RAM
    ram = psutil.virtual_memory()
    used = ram.used / 1024**3
    total_ram = ram.total / 1024**3
    percent = ram.percent
    logger.info("%sRAM  — used: %.2fGB / %.2fGB (%.1f%%)", prefix, used, total_ram, percent)
