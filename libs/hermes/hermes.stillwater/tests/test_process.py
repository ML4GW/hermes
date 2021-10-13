import time

import pytest

from hermes.stillwater.process import Pipeline, PipelineProcess
from hermes.stillwater.utils import ExceptionWrapper


class ErrorProcess(PipelineProcess):
    def get_package(self):
        return 1

    def process(self, package):
        raise RuntimeError("whoops!")


class RangeProcess(PipelineProcess):
    def __init__(self, N, *args, **kwargs):
        self.N = N
        self.n = 0
        super().__init__(*args, **kwargs)

    def get_package(self):
        if self.n == self.N:
            raise StopIteration
        else:
            n = self.n + 0
            self.n += 1
            return n


class AddOneProcess(PipelineProcess):
    def get_package(self):
        return super().get_package() + 1


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


@pytest.mark.depends(on=["test_process"])
def test_pipeline():
    generator = RangeProcess(100, name="generator")
    returner = PipelineProcess(name="returner")

    # validate basic pipeline behavior
    returner.in_q = generator.out_q
    with Pipeline([generator, returner]) as pipeline:
        i = 0
        for j in pipeline:
            assert j == i
            i += 1

    # now validate pipeline piping into process
    generator = RangeProcess(100, name="generator")
    returner = PipelineProcess(name="returner")
    one_adder = AddOneProcess(name="add_one")

    returner.in_q = generator.out_q
    pipeline = Pipeline([generator, returner])
    with pipeline >> one_adder as pipeline:
        i = 1
        for j in pipeline:
            assert j == i
            i += 1

    # now validate pipeline to pipeline piping
    generator = RangeProcess(100, name="generator")
    returner = PipelineProcess(name="returner")
    one_adder = AddOneProcess(name="add_one")
    generator.out_q = returner.in_q
    returner.out_q = one_adder.in_q
    pipeline_1 = Pipeline([generator, returner, one_adder])

    one_adder_2 = AddOneProcess(name="add_one_2")
    returner_2 = PipelineProcess(name="returner_2")
    one_adder_2.out_q = returner_2.in_q
    pipeline_2 = Pipeline([one_adder_2, returner_2])

    with pipeline_1 >> pipeline_2 as pipeline:
        i = 2
        for j in pipeline:
            assert j == i
            i += 1

    # now make sure that an error gets raised if
    # we try to pipe to anything else
    with pytest.raises(TypeError):
        returner = Pipeline(name="returner")
        returner >> "bad type"


@pytest.mark.depends(on=["test_pipeline"])
def test_process_piping():
    # test the piping system for connecting
    # one process to the another, using a process
    # that just iterates through a range and another
    # that just returns whatever it's passed unchanged
    generator = RangeProcess(100, name="generator")
    returner = PipelineProcess(name="returner")
    with generator >> returner as pipeline:
        i = 0
        for j in pipeline:
            assert i == j
            i += 1

    # now test process to pipeline piping
    generator = RangeProcess(100, name="generator")
    returner = PipelineProcess(name="returner")
    one_adder = AddOneProcess(name="add_one")
    pipeline = returner >> one_adder

    with generator >> pipeline as pipeline:
        i = 1
        for j in pipeline:
            assert i == j
            i += 1

    with pytest.raises(TypeError):
        returner = PipelineProcess(name="returner")
        returner >> "bad type"
