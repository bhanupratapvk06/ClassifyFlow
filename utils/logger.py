import os
import logging

# Ensure logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)


def create_logger(file_name):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_log_path = os.path.join(log_dir, f'{file_name}.log')
    file_handler = logging.FileHandler(file_log_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
