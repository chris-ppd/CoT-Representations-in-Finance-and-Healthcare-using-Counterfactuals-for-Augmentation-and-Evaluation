import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setup_logging(
    run_name: str,  # e.g. "cot_generation_ld1"
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:

    log_dir = PROJECT_ROOT / Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name}_{timestamp}.log"

    # Format: [2026-05-10 14:32:01] [INFO] [src.cot.cot_generator] Starting batch 3
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — same output you have now
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # File handler — persists everything
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # capture everything in file

    # Root logger — catches all loggers in your codebase
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Silence noisy third-party libraries
    for noisy_lib in ["transformers", "huggingface_hub", "httpcore", "httpx"]:
        logging.getLogger(noisy_lib).setLevel(logging.ERROR)

    root_logger.info(f"Logging initialised — file: {log_file}")

    return root_logger
