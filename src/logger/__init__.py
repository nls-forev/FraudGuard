import os
import logging

from from_root import from_root

from logging.handlers import RotatingFileHandler

from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

log_dir_path = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)


def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )

    # Set log rotation
    # A log file nearing 5 MB will automatically rotate to a new file.
    fileHandler = RotatingFileHandler(
        log_file_path,
        backupCount=BACKUP_COUNT,
        maxBytes=MAX_LOG_SIZE,
    )
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(logging.DEBUG)

    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)


configure_logger()
