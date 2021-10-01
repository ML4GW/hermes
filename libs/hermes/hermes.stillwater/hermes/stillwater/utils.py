import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from tblib import pickling_support

if TYPE_CHECKING:
    import numpy as np


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


class Throttle:
    def __init__(self, target_rate: float, alpha: float = 0.9):
        self.target_rate = target_rate
        self.alpha = alpha
        self.unset()

    def unset(self):
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
        self._last_time = time.time()
        self._n += 1

        diff = (1 / self.rate) - (1 / self.target_rate)
        self._delta = self._delta + (1 - self.alpha) * diff

    def __enter__(self):
        self._start_time = self._last_time = time.time()
        return self

    def __exit__(self, *exc_args):
        self.unset()

    def throttle(self):
        while (time.time() - self._last_time) < self.sleep_time:
            time.sleep(1e-6)
        self.update()


@dataclass
class Package:
    x: "np.ndarray"
    t0: float
    request_id: Optional[int] = None
    sequence_id: Optional[int] = None
    sequence_start: Optional[bool] = None
    sequence_end: Optional[bool] = None
