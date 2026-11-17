"""Logging configuration with Rich handler for pretty console output.

Provides a shared Rich console and logging setup so that all modules can use
standard `logging` calls routed through a `RichHandler`.
"""

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

console = Console(stderr=True, force_terminal=True)
"""Shared Rich console for logging and progress bars (writes to stderr).

Uses `force_terminal=True` so that progress bars render even in non-TTY
environments (e.g. Slurm log files).
"""


def setup_logging() -> None:
    """Configure the `genret` logger with a RichHandler.

    Reads the `LOG_LEVEL` environment variable (default `INFO`) and installs
    a `RichHandler` on the `genret` logger that writes to :data:`console`.
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    handler = RichHandler(
        console=console,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    genret_logger = logging.getLogger("genret")
    genret_logger.setLevel(level)
    genret_logger.addHandler(handler)
    genret_logger.propagate = False
