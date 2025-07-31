# ================================================================
#  src/utils.py
#
#  Utility functions for the project:
## 1. get_logger: Configures and returns a logger instance.
## 2. add_project_root_to_path: Adds the project root directory to sys.path.

# Author: Flavio Aguirre
# Date: 2025-07-31

# ================================================================
# Import necessary libraries
# ================================================================

import logging
import os
import sys

from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

# ========================================================================
# 1. get_logger
# ========================================================================
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger instance.

    Parameters
    ----------
    name : str
        Logger name, usually __name__.
    level : int
        Logging level. Default: INFO.

    Returns
    -------
    logging.Logger
        Configured logger with console and file output.
        
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        os.makedirs("logs", exist_ok=True)
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = RotatingFileHandler("logs/project.log", maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



# ========================================================================
# 2. add_project_root_to_path
# ========================================================================
def add_project_root_to_path() -> None:
    """
    Add the project root directory to sys.path if not already present.

    Parameters
    ----------
    None

    Returns
    -------
    None
    
    """
    logger = get_logger(__name__)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    if not os.path.isdir(project_root):
        raise FileNotFoundError(f"Project root not found: {project_root}")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added project root to sys.path: {project_root}")
    else:
        logger.debug(f"Project root already in sys.path: {project_root}")



# ========================================================================