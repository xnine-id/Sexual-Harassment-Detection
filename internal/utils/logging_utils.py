import logging
import os
import sys
from typing import Optional


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        try:
            if (hasattr(self, "_isatty") and self._isatty) or getattr(
                self, "_isatty_checked", False
            ) is False:
                self._isatty = sys.stderr.isatty() or sys.stdout.isatty()
                self._isatty_checked = True
            if getattr(self, "_isatty", False) and os.getenv("NO_COLOR") is None:
                color = self.COLORS.get(original)
                if color:
                    record.levelname = f"{color}{original}{self.RESET}"
            return super().format(record)
        finally:
            record.levelname = original


def setup_logging(
    level_name: Optional[str] = None,
    use_color: bool = True,
    fmt: str = "%(levelname)s [%(name)s] - %(message)s",
) -> None:
    level_str = (level_name or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stdout)
    if use_color:
        handler.setFormatter(ColorFormatter(fmt))
    else:
        handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
