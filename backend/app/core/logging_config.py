import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.config import get_settings

settings = get_settings()


def setup_logging() -> logging.Logger:
    """Configure structured logging with console + rotating file handler."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Rotating file handler (10 MB × 5 backups)
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "chromadb", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("app")


logger = setup_logging()
