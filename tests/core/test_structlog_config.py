"""Unit tests for configure_logging, get_logger, and _ensure_log_dir."""

import logging
import logging.handlers
from pathlib import Path

import pytest

from src.core.structlog_config import configure_logging, get_logger, _ensure_log_dir


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Restore root logger state after each test."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    for h in root.handlers:
        if h not in original_handlers:
            h.close()
    root.handlers[:] = original_handlers
    root.setLevel(original_level)


# ---------------------------------------------------------------------------
# configure_logging — handler count and types
# ---------------------------------------------------------------------------


def test_configure_logging_adds_exactly_two_handlers(tmp_path):
    """configure_logging() must install exactly one console and one file handler."""
    configure_logging(log_path=str(tmp_path / "app.log"))
    assert len(logging.getLogger().handlers) == 2


def test_configure_logging_installs_file_handler(tmp_path):
    """configure_logging() must include a TimedRotatingFileHandler."""
    configure_logging(log_path=str(tmp_path / "app.log"))
    file_handlers = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.handlers.TimedRotatingFileHandler)
    ]
    assert len(file_handlers) == 1


def test_configure_logging_installs_console_handler(tmp_path):
    """configure_logging() must include a StreamHandler for console output."""
    configure_logging(log_path=str(tmp_path / "app.log"))
    console_handlers = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.handlers.TimedRotatingFileHandler)
    ]
    assert len(console_handlers) == 1


# ---------------------------------------------------------------------------
# configure_logging — log directory creation
# ---------------------------------------------------------------------------


def test_configure_logging_creates_nested_log_directory(tmp_path):
    """configure_logging() must create all missing parent directories for the log file."""
    log_path = tmp_path / "nested" / "logs" / "app.log"
    configure_logging(log_path=str(log_path))
    assert log_path.parent.exists()


# ---------------------------------------------------------------------------
# configure_logging — string log levels
# ---------------------------------------------------------------------------


def test_configure_logging_accepts_string_log_levels(tmp_path):
    """configure_logging() must accept string values like 'DEBUG' and 'INFO' for levels."""
    configure_logging(
        console_level="INFO",
        file_level="DEBUG",
        log_path=str(tmp_path / "app.log"),
    )
    assert len(logging.getLogger().handlers) == 2


# ---------------------------------------------------------------------------
# configure_logging — json_console flag
# ---------------------------------------------------------------------------


def test_configure_logging_json_console_mode_does_not_raise(tmp_path):
    """configure_logging(json_console=True) must succeed without errors."""
    configure_logging(log_path=str(tmp_path / "app.log"), json_console=True)
    assert len(logging.getLogger().handlers) == 2


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


def test_get_logger_returns_a_logger():
    """get_logger() must return a non-None logger instance."""
    logger = get_logger("test.module")
    assert logger is not None


def test_get_logger_uses_provided_name():
    """get_logger() called with different names must return usable loggers."""
    logger_a = get_logger("module.a")
    logger_b = get_logger("module.b")
    assert logger_a is not None
    assert logger_b is not None


# ---------------------------------------------------------------------------
# _ensure_log_dir
# ---------------------------------------------------------------------------


def test_ensure_log_dir_creates_parent_directories(tmp_path):
    """_ensure_log_dir() must create all nested parent directories."""
    log_path = str(tmp_path / "a" / "b" / "c" / "app.log")
    _ensure_log_dir(log_path)
    assert Path(log_path).parent.exists()


def test_ensure_log_dir_is_idempotent(tmp_path):
    """_ensure_log_dir() must not raise when the directory already exists."""
    log_path = str(tmp_path / "logs" / "app.log")
    _ensure_log_dir(log_path)
    _ensure_log_dir(log_path)
    assert Path(log_path).parent.exists()
