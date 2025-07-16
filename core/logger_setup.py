import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name, log_file='logs/bot.log', level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
