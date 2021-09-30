import time

import pytest

from hermes.stillwater.process import PipelineProcess
from hermes.stillwater.utils import ExceptionWrapper


class ErrorProcess(PipelineProcess):
    def get_package(self):
        return 1

    def process(self, package):
        raise RuntimeError("whoops!")


def test_process(throttle_tol=0.15):
    # test that process starts when context enters
    with PipelineProcess(name="test_process") as process:
        assert process.is_alive()

    # now that the context is exited, the
    # process should have closed and accessing
    # its standard attributes should raise a ValueError
    with pytest.raises(ValueError):
        process.is_alive()

    # test iterator behavior, including that
    # StopIteration raise will break out of loop
    with PipelineProcess(name="test_process") as process:
        for i in range(10):
            process.in_q.put(i)
        process.in_q.put(ExceptionWrapper(StopIteration()))

        i = 0
        for j in process:
            assert i == j
            i += 1

    # make sure that the throttling behavior works as expected
    with PipelineProcess(name="throttle_process", rate=1000) as process:
        start_time = time.time()
        for i in range(2000):
            process.in_q.put(i)
        process.in_q.put(ExceptionWrapper(StopIteration()))

        # wait for process to reach the StopIteration
        for _ in process:
            continue

        # make sure that the time that has passed
        # is roughly in range with what we'd expect
        elapsed = time.time() - start_time
        expected_time = (i + 1) / process.throttle.target_rate
        hi = (1 + throttle_tol) * expected_time
        lo = (1 - throttle_tol) * expected_time
        assert lo <= elapsed <= hi

    # make sure that the error in the process gets
    # bubbled up to the main process via the iterator
    with pytest.raises(RuntimeError):
        with ErrorProcess(name="error_process") as process:
            next(iter(process))
