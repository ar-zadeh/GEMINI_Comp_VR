"""
vr_agent/logger.py
------------------
AgentLogger class and get_logger() singleton.
"""

import logging
import json
from pathlib import Path
from datetime import datetime

from .config import LOG_DIR


class AgentLogger:
    """Simple logger setup: writes to file + stdout."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{self.session_id}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("VRAgent")

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def action(self, tool, args, result):
        self.info(f"[ACTION] {tool}({json.dumps(args)}) -> {str(result)[:200]}")


# ── Singleton ─────────────────────────────────────────────────────────────────
_logger = None


def get_logger() -> AgentLogger:
    global _logger
    if not _logger:
        _logger = AgentLogger(LOG_DIR)
    return _logger
