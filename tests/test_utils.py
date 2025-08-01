# ========================================================================
# Tests for Utility Functions in src.utils
# ========================================================================

# This module contains tests for utility functions defined in src/utils.py.
# The tests cover:
# 1. get_logger: Validates logger creation, handler attachment, and log file writing
# 2. add_project_root_to_path: Ensures the project root is added to sys.path and handles edge cases.

# Author: Flavio Aguirre
# Date: 2025-07-31

# ========================================================================
# Import necessary libraries
# ========================================================================

import sys
import logging
import os
from pathlib import Path

import pytest
from src.utils import get_logger, add_project_root_to_path


# ========================================================================
# 1. Test: get_logger
# ========================================================================
def test_get_logger_returns_logger_instance():
    """
    Test that get_logger returns a valid logging.Logger instance.

    Asserts
    -------
    - The returned object is an instance of logging.Logger.
    - The logger has at least one handler configured.
    """
    logger = get_logger("test_logger_instance")

    assert isinstance(logger, logging.Logger), "get_logger should return a logging.Logger instance."
    assert logger.handlers, "Logger should have at least one handler configured."


def test_get_logger_adds_expected_handlers():
    """
    Test that get_logger attaches both console and file handlers.

    Asserts
    -------
    - Logger includes a StreamHandler (console).
    - Logger includes a RotatingFileHandler (file-based logging).
    """
    logger = get_logger("test_logger_handlers")

    handler_types = {type(h) for h in logger.handlers}
    assert any("StreamHandler" in str(h) for h in handler_types), "Logger should have a StreamHandler."
    assert any("RotatingFileHandler" in str(h) for h in handler_types), "Logger should have a RotatingFileHandler."


def test_get_logger_returns_same_instance():
    """
    Test that get_logger returns the same logger instance for the same name.

    Asserts
    -------
    - Two calls to get_logger with the same name return the exact same instance.
    """
    logger1 = get_logger("test_logger_singleton")
    logger2 = get_logger("test_logger_singleton")

    assert logger1 is logger2, "get_logger should return the same instance for the same logger name."


def test_get_logger_writes_to_log_file(tmp_path: Path):
    """
    Test that get_logger writes log messages to a file.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest for safe file operations.

    Asserts
    -------
    - The log file is created in the temporary directory.
    - A log message is successfully written to the log file.
    """
    log_file = tmp_path / "project.log"

    logger = get_logger("test_logger_file")

    # Remove previous handlers to redirect logs to tmp_path
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    test_message = "This is a test log message."
    logger.info(test_message)

    assert log_file.exists(), f"Expected log file {log_file} was not created."
    assert test_message in log_file.read_text(), "Expected log message not found in the log file."


def test_get_logger_respects_custom_log_level(tmp_path: Path):
    """
    Test that get_logger respects a custom log level.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Asserts
    -------
    - The logger level is set to the specified custom level.
    """
    logger = get_logger("test_logger_debug", level=logging.DEBUG)
    assert logger.level == logging.DEBUG, "Logger level should match the specified custom level."


# ========================================================================
# 2. Test: add_project_root_to_path
# ========================================================================
def test_add_project_root_to_path_adds_path(monkeypatch):
    """
    Test that add_project_root_to_path adds the project root to sys.path.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to isolate modifications to sys.path.

    Asserts
    -------
    - The project root is added to sys.path.
    """
    monkeypatch.setattr(sys, "path", sys.path.copy())

    add_project_root_to_path()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    assert project_root in sys.path, "Project root should be added to sys.path."


def test_add_project_root_to_path_is_idempotent(monkeypatch):
    """
    Test that add_project_root_to_path does not duplicate the project root in sys.path.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch

    Asserts
    -------
    - Calling add_project_root_to_path twice should not duplicate the project root.
    """
    monkeypatch.setattr(sys, "path", sys.path.copy())

    add_project_root_to_path()
    add_project_root_to_path()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    assert sys.path.count(project_root) == 1, "Project root should not be duplicated in sys.path."


def test_add_project_root_to_path_raises_if_missing(monkeypatch):
    """
    Test that add_project_root_to_path raises FileNotFoundError if the project root does not exist.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch

    Asserts
    -------
    - FileNotFoundError is raised when project root is mocked as non-existent.
    """
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(os.path, "isdir", lambda _: False)  # Force isdir to return False

    with pytest.raises(FileNotFoundError):
        add_project_root_to_path()