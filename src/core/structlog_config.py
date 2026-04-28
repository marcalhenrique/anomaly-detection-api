import logging
import logging.handlers
import sys
from pathlib import Path

import structlog


def configure_logging(
    console_level: int | str = logging.DEBUG,
    file_level: int | str = logging.DEBUG,
    log_path: str = "log/app.log",
    json_console: bool = False,
) -> None:
    """Configure structlog with console (colored) and file (JSON) handlers.

    Args:
        console_level: Log level for console output.
        file_level: Log level for file output.
        log_path: Path to the rotating log file.
        json_console: If True, console output is also JSON (useful in production).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    console_renderer = (
        structlog.processors.JSONRenderer()
        if json_console
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            console_renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    if isinstance(console_level, str):
        console_level = logging.getLevelName(console_level.upper())
    if isinstance(file_level, str):
        file_level = logging.getLevelName(file_level.upper())

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    _ensure_log_dir(log_path)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        backupCount=30,
        interval=1,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    root.addHandler(file_handler)


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("indexing started", collection="papers", chunks=42)

    For request correlation, bind context vars in middleware:
        structlog.contextvars.bind_contextvars(request_id="abc-123")
    """
    return structlog.get_logger(name)


def _ensure_log_dir(log_path: str) -> None:
    """Create the log directory if it does not exist."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
