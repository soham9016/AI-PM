"""
Run-path logger.

Records which agents executed, in what order, independent of the graph's
own `run_path` state field — useful for debugging while the state schema
and routing are still being built, and as a live console trace once the
graph is running.
"""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("business_copilot.run_path")


class RunPathLogger:
    """Tracks the ordered sequence of agent executions for a single run."""

    def __init__(self) -> None:
        self._path: list[dict] = []

    def record(self, agent_name: str, **metadata) -> None:
        entry = {"agent": agent_name, "timestamp": time.time(), **metadata}
        self._path.append(entry)
        logger.info("[run_path] %s %s", agent_name, metadata or "")

    @property
    def path(self) -> list[str]:
        """Ordered list of agent names only, e.g. ['structurer', 'hypothesis', ...]."""
        return [entry["agent"] for entry in self._path]

    @property
    def full_path(self) -> list[dict]:
        """Ordered list of full entries (agent + timestamp + any metadata)."""
        return list(self._path)

    def reset(self) -> None:
        self._path.clear()


# Module-level singleton — import and use directly from agent node functions:
#   from utils.logger import run_logger
#   run_logger.record("structurer")
run_logger = RunPathLogger()
