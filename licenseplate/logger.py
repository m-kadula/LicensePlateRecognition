from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


def get_standard_logger(name: str, output) -> logging.Logger:
    detection_logger = logging.getLogger(name)
    detection_logger.setLevel(logging.DEBUG)

    detection_logger_handler = logging.StreamHandler(output)
    detection_logger_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    detection_logger_handler.setFormatter(formatter)

    detection_logger.addHandler(detection_logger_handler)
    return detection_logger


def get_rotating_logger(
    name: str,
    log_dir: Path,
    log_filename: str,
    level=logging.DEBUG,
    max_bytes: int = 1024 * 1024,
    backup_count: int = 5,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    log_dir.mkdir(exist_ok=True, parents=True)

    log_path = log_dir / log_filename

    handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
