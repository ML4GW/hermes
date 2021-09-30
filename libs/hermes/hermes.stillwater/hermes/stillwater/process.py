import multiprocessing as mp
import sys
import time
from contextlib import nullcontext
from queue import Empty
from typing import Optional

from hermes.stillwater.logging import listener, logger
from hermes.stillwater.utils import ExceptionWrapper, Throttle


class PipelineProcess(mp.Process):
    def __init__(
        self, name: str, rate: Optional[float] = None, join_timeout: float = 10
    ) -> None:
        self._pause_event = mp.Event()
        self._stop_event = mp.Event()

        self.in_q = mp.Queue()
        self.out_q = mp.Queue()

        if rate is not None:
            self.throttle = Throttle(rate)
        else:
            self.throttle = nullcontext()

        self.join_timeout = join_timeout
        super().__init__(name=name)

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def cleanup(self, exc):
        self.logger.error(f"Encountered {exc.__class__.__name__}")
        self.out_q.put(ExceptionWrapper(exc))

        self.stop()

    def _impatient_get(self, q):
        while True:
            try:
                item = q.get_nowait()
            except Empty:
                if self.stopped:
                    break
                time.sleep(1e-6)
            else:
                if isinstance(item, ExceptionWrapper):
                    item.reraise()
                return item

    def get_package(self):
        return self._impatient_get(self.in_q)

    def process(self, package):
        self.out_q.put(package)

    def run(self) -> None:
        exitcode = 0
        try:
            self.logger = listener.add_process(self)
            self.logger.warning("hey!")

            with self.throttle:
                while not self.stopped:
                    inputs = self.get_package()
                    if inputs is not None:
                        try:
                            self.process(*inputs)
                        except TypeError:
                            self.process(inputs)

                        if not isinstance(self.throttle, nullcontext):
                            self.throttle.throttle()
        except Exception as e:
            self.cleanup(e)
            exitcode = 1
        finally:
            self.logger.debug("Target completed")
            listener.queue.join()
            sys.exit(exitcode)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type_, value, traceback):
        if not self.stopped:
            self.stop()

        # stop the listener before exiting so
        # that it doesn't try to poll connections
        # from dead threads
        if type_ is not None and listener._thread is not None:
            listener.stop()

        self.join(self.join_timeout)
        if self.exitcode is None:
            logger.warning(f"Process {self.name} couldn't close gracefully")
            self.terminate()
            time.sleep(1)
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self._impatient_get(self.out_q)
