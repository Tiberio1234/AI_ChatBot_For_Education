import logging
import os
from datetime import datetime

def setup_logger(name):
    # Create a custom logger
    logger = logging.getLogger(name if name != None else __name__)
    logger.setLevel(logging.DEBUG)  # Set minimum log level
    
    # Create formatter that includes timestamp, filename, and line number
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler and set formatter
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    # Add handler to the logger
    logger.addHandler(ch)
    
    return logger