"""
Logging configuration for the Anki pipelines.

Provides a centralized logging setup with human-readable output and structured context fields.
"""
import logging
import sys
from typing import Optional


class ContextFormatter(logging.Formatter):
    """Custom formatter that adds structured context fields to log messages.

    Supports extra fields passed via logger.info("msg", extra={...})
    Format: timestamp [LEVEL] logger_name: message | key1=value1 key2=value2
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
        'GRAY': '\033[90m',       # Gray for extra fields
    }

    def format(self, record: logging.LogRecord) -> str:
        # Format the base message
        base_msg = super().format(record)

        # Add color to log level
        levelname = record.levelname
        if levelname in self.COLORS:
            # Replace [LEVEL] with colored version
            colored_level = f"{self.COLORS[levelname]}[{levelname}]{self.COLORS['RESET']}"
            base_msg = base_msg.replace(f"[{levelname}]", colored_level)

        # Add extra context fields if present
        extra_fields = []
        for key, value in record.__dict__.items():
            # Skip standard logging attributes and None values
            if key not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName', 'relativeCreated',
                'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                'asctime', 'getMessage', 'taskName'
            ] and value is not None:
                extra_fields.append(f"{key}={value}")

        if extra_fields:
            # Gray color for extra fields
            extra_str = f"{self.COLORS['GRAY']} | {' '.join(extra_fields)}{self.COLORS['RESET']}"
            return f"{base_msg}{extra_str}"
        return base_msg


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the entire application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Use DEBUG to see LLM prompts/responses and detailed execution flow.
               Use INFO for normal operation (default).
    
    Example:
        >>> from anki.common.logging_config import setup_logging
        >>> setup_logging("DEBUG")  # Enable detailed logging
        >>> setup_logging("INFO")   # Normal operation
    """
    # Convert string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter with human-readable format
    formatter = ContextFormatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger for 'anki' namespace
    root_logger = logging.getLogger('anki')
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug("Detailed debug info")
        >>> logger.info("Progress update", extra={"processed": 10, "total": 100})
    """
    return logging.getLogger(name)

