import logging
import os
from datetime import datetime

## Creating log directory
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)

## Creating log file name with timestamp
FILE_NAME = f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE = os.path.join(LOGS_DIR, FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger