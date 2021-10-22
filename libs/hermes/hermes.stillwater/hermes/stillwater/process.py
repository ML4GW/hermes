import multiprocessing as mp
import sys
import time
from queue import Empty
from typing import TYPE_CHECKING, Optional

from hermes.stillwater.logging import listener, logger
from hermes.stillwater.utils import ExceptionWrapper, Throttle

if TYPE_CHECKING:
    from queue import Queue

    from hermes.stillwater.utils import Package


class PipelineProcess(mp.Process):
    def __init__(
        self, name: str, rate: Optional[float] = None, join_timeout: float = 10
    ) -> None:
        self._pause_event = mp.Event()
        self._stop_event = mp.Event()

        self.in_q = mp.Queue()
        self.out_q = mp.Queue()

        # build a throttle to use during the target process
        # to limit ourselves to the target rate if we passed
        # one, otherwise we'll just iterate infinitely. In
        # either case, use the stop event's set status to
        # indicate when the loop should be interrupted
        if rate is not None:
            self.throttle = Throttle(rate, condition=self._stop_event.is_set)
        else:
            self.throttle = iter(self._stop_event.is_set, True)

        self.join_timeout = join_timeout
        self.logger = None
        super().__init__(name=name)

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def cleanup(self, exc: Exception) -> None:
        """Gracefully clean up the process if an exception is encountered"""

        if self.logger is not None:
            self.logger.error(f"Encountered {exc.__class__.__name__}: {exc}")
        self.out_q.put(ExceptionWrapper(exc))

        self.stop()

    def _impatient_get(self, q: "Queue") -> "Package":
        """Wait forever to get an object from a queue

        Gets and item from a queue in a way that
        waits forever without blocking so that
        errors that get bubbled up can interrupt
        appropriately. Also checks to see if upstream
        processes have passed an exception and raises
        them so that the traceback is maintained.

        Args:
            q:
                The queue to get from
        """

        while True:
            try:
                item = q.get_nowait()
            except Empty:
                time.sleep(1e-6)
            else:
                if isinstance(item, ExceptionWrapper):
                    item.reraise()
                elif item == StopIteration or isinstance(item, StopIteration):
                    raise StopIteration
                return item

    def get_package(self) -> "Package":
        return self._impatient_get(self.in_q)

    def process(self, package: "Package") -> None:
        self.out_q.put(package)

    def run(self) -> None:
        exitcode = 0
        try:
            # create a multiprocessing logger that
            # write logs to the main process for handling
            self.logger = listener.add_process(self)

            # run everything in a throttle context in
            # case we want to rate control everything
            for _ in self.throttle:
                # try to get the next package, and process
                # it if there's anything to process
                inputs = self.get_package()
                if inputs is not None:

                    # try passing a starmap so that subclasses
                    # don't always have to return lists, but
                    # otherwise call the downstream process
                    # normally
                    if not isinstance(inputs, dict):
                        try:
                            self.process(*inputs)
                        except TypeError:
                            self.process(inputs)
                    else:
                        self.process(inputs)

        except Exception as e:
            # pass on any exceptions to downstream processes
            # and set the exitcode to indicate an error
            # TODO: should we do a special case for StopIterations?
            self.cleanup(e)
            exitcode = 1
        finally:
            # send one last log to the main process then
            # close the queue and wait for the thread to join
            self.logger.debug("Target completed")
            listener.queue.close()
            listener.queue.join_thread()

            # exit the process with the indicated code
            sys.exit(exitcode)

    def __enter__(self) -> "PipelineProcess":
        self.start()
        return self

    def __exit__(self, type_, value, traceback) -> None:
        if not self.stopped:
            self.stop()

        # stop the listener before exiting so
        # that it doesn't try to poll connections
        # from dead threads
        if type_ is not None and listener._thread is not None:
            listener.stop()

        # try to join the process if we can
        self.join(self.join_timeout)
        if self.exitcode is None:
            # if the process is still running after the wait
            # time, terminate it and log a warning
            logger.warning(
                f"Process {self.name} couldn't join gracefully. Terminating"
            )
            self.terminate()
            time.sleep(1)
        else:
            logger.debug(f"Process {self.name} joined gracefully")

        # close the process
        self.close()

        # clear and close the input queue
        # to kill the daemon thread
        logger.debug(f"Clearing input queue for process {self.name}")
        while True:
            try:
                self.in_q.get_nowait()
            except Empty:
                break
        logger.debug(f"Input queue for process {self.name} cleared")
        self.in_q.close()

    def __iter__(self) -> "PipelineProcess":
        return self

    def __next__(self) -> "Package":
        return self._impatient_get(self.out_q)

    def __rshift__(self, child) -> "PipelineProcess":
        if isinstance(child, Pipeline):
            child.processes[0].in_q = self.out_q
            processes = [self] + child.processes
            return Pipeline(processes)
        elif isinstance(child, PipelineProcess):
            child.in_q = self.out_q
            return Pipeline([self, child])
        else:
            raise TypeError(
                "Unsupported operand type(s) for >> "
                "PipelineProcess and {}".format(type(child))
            )


class Pipeline:
    def __init__(self, processes):
        self.processes = processes

    def __enter__(self):
        for p in self.processes:
            p.__enter__()
        return self

    def __exit__(self, *exc_args):
        for p in self.processes:
            p.__exit__(*exc_args)

    def __iter__(self):
        return iter(self.processes[-1])

    def __rshift__(self, child):
        if isinstance(child, PipelineProcess):
            child.in_q = self.processes[-1].out_q
            processes = self.processes + [child]
            return Pipeline(processes)
        elif isinstance(child, Pipeline):
            child.processes[0].in_q = self.processes[-1].out_q
            processes = self.processes + child.processes
            return Pipeline(processes)
        else:
            raise TypeError(
                "Unsupported operand type(s) for >> "
                "Pipeline and {}".format(type(child))
            )
