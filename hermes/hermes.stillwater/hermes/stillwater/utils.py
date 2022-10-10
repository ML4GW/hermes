import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

from tblib import pickling_support


@pickling_support.install
class ExceptionWrapper(Exception):
    def __init__(self, msg: Exception) -> None:
        self.exc = msg
        _, __, self.tb = sys.exc_info()
        super().__init__(str(msg))

    def reraise(self) -> None:
        raise self.exc.with_traceback(self.tb)

    def __str__(self):
        return str(self.exc)

    def __repr__(self):
        return str(self.exc)


@dataclass
class Throttle:
    target_rate: float
    alpha: float = 0.9
    condition: Optional[Callable] = None
    update_every: int = 100

    def __post_init__(self):
        self._n = 0
        self._delta = 0
        self._start_time = None
        self._last_time = None

    @property
    def rate(self):
        if self._start_time is None:
            return None
        return self._n / (time.time() - self._start_time)

    @property
    def sleep_time(self):
        return (1 / self.target_rate) - self._delta

    def update(self):
        diff = (1 / self.rate) - (1 / self.target_rate)
        self._delta = self._delta + (1 - self.alpha) * diff

    def __iter__(self):
        self._start_time = self._last_time = time.time()
        return self

    def __next__(self):
        if self.condition is not None and self.condition():
            self.__post_init__()
            raise StopIteration

        while (time.time() - self._last_time) < self.sleep_time:
            time.sleep(1e-6)

        self._n += 1
        self._last_time = time.time()
        if self._n % self.update_every == 0:
            self.update()
