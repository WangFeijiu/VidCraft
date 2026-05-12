"""Task executor — ThreadPoolExecutor wrapper with cancel tracking."""
from concurrent.futures import ThreadPoolExecutor

from loguru import logger


class TaskExecutor:
    def __init__(self, max_workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vs_worker")
        self.cancel_set: set[str] = set()

    def submit(self, fn, *args, **kwargs):
        logger.debug(f"Submitting task: {fn.__name__}")
        return self._pool.submit(fn, *args, **kwargs)

    def cancel(self, name: str) -> None:
        self.cancel_set.add(name)

    def is_cancelled(self, name: str) -> bool:
        return name in self.cancel_set

    def clear_cancel(self, name: str) -> None:
        self.cancel_set.discard(name)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
