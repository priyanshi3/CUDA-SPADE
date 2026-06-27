"""
alerts.py — Alert dispatcher.

Writes alerts to a rotating log file and keeps a recent-alert list
that the dashboard reads on every refresh.
"""

import os
import logging
from datetime import datetime


class AlertDispatcher:
    def __init__(self, log_file: str, max_recent: int = 20):
        self._recent = []
        self._max    = max_recent
        self._logger = self._make_logger(log_file)

    @staticmethod
    def _make_logger(log_file: str) -> logging.Logger:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        logger = logging.getLogger("spade.alerts")
        if not logger.handlers:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(fh)
            logger.setLevel(logging.INFO)
        return logger

    def send(self, sensor_name: str, severity: str, message: str):
        ts    = datetime.now().strftime("%H:%M:%S")
        entry = f"{ts} {severity:<6} {sensor_name:<22} | {message}"
        self._recent.append(entry)
        if len(self._recent) > self._max:
            self._recent.pop(0)
        if severity == "ALERT":
            self._logger.warning(f"{sensor_name} | {message}")
        else:
            self._logger.info(f"{sensor_name} | {message}")

    def get_recent(self) -> list[str]:
        """Return most-recent entries first (for top-of-panel display)."""
        return list(reversed(self._recent))
