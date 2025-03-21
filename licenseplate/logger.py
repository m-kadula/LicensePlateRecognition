import logging


def get_logger(name: str, output) -> logging.Logger:
    detection_logger = logging.getLogger(name)
    detection_logger.setLevel(logging.DEBUG)

    detection_logger_handler = logging.StreamHandler(output)
    detection_logger_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    detection_logger_handler.setFormatter(formatter)

    detection_logger.addHandler(detection_logger_handler)
    return detection_logger
