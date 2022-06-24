import atexit
import logging
import os
from logging.handlers import QueueListener
from multiprocessing import JoinableQueue

logger = logging.getLogger("hermes.stillwater")
logger.addHandler(logging.NullHandler())


class LogListener(QueueListener):
    def __init__(self, queue, *handlers, respect_handler_level=False):
        super().__init__(
            queue, *handlers, respect_handler_level=respect_handler_level
        )

    def add_process(self, process):
        pid = os.getpid()
        logger = logging.getLogger(
            "hermes.stillwater.{}.{}".format(process.name, pid)
        )
        logger.addHandler(logging.handlers.QueueHandler(self.queue))
        return logger


log_q = JoinableQueue(-1)
listener = LogListener(
    log_q, logging.NullHandler(), respect_handler_level=True
)
listener.start()


def shutdown():
    if listener._thread is not None:
        listener.stop()


atexit.register(shutdown)
