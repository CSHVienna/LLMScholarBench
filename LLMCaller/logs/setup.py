import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(run_dir):
    logger = logging.getLogger('experiment_runner')
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = RotatingFileHandler(
        os.path.join(run_dir, 'experiment_runner.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    console_format = logging.Formatter(log_format)
    file_format = logging.Formatter(log_format)
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger