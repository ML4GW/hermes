import multiprocessing as mp
import random
import time
from typing import Optional, Tuple

import numpy as np
import tritonclient.grpc as triton

from hermes.stillwater.process import PipelineProcess
from hermes.stillwater.utils import Package


def _raise_no_match(self, attr, first, second):
    raise ValueError(
        "Package {}s don't all match. Found {}s {} and {}".format(
            attr, attr, first, second
        )
    )


class InferenceClient(PipelineProcess):
    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: int,
        profile: bool = False,
        batch_size: int = 1,
        rate: Optional[float] = None,
        join_timeout: float = 10,
        name: Optional[str] = None,
    ):
        try:
            client = triton.InferenceServerClient(url)

            if not client.is_server_live():
                raise RuntimeError(f"Server at url {url} isn't live")
            if not client.is_model_ready(model_name):
                raise RuntimeError(
                    f"Model {model_name} isn't ready at server url {url}"
                )
        except triton.InferenceServerException:
            raise RuntimeError(f"Couldn't connect to server at {url}")

        self.client = client
        self.url = url
        self.model_name = model_name
        self.model_version = model_version

        self.inputs, self.states = self.build_inputs(batch_size)

        self.message_start_times = {}
        self.profile = profile
        if self.profile:
            self._profile_q = mp.Queue()
            self._start_times = {}

        super().__init__(name, rate, join_timeout)

    def build_inputs(self, batch_size):
        config = self.client.get_model_config(self.model_name).config
        metadata = self.client.get_model_metadata(self.model_name)

        states, inputs = [], []
        for config_input, metadata_input in zip(config.input, metadata.inputs):
            shape = [i if i > 0 else batch_size for i in metadata_input.shape]
            input = triton.InferInput(
                name=metadata_input.name,
                shape=shape,
                datatype=metadata_input.datatype,
            )
            for step in config.ensemble_scheduling.step:
                # see if this input corresponds to the input
                # for a snapshotter model. TODO: come up
                # with a better way of identifying snapshotter
                # models than by using the name
                if (
                    len(step.input_map) == 1
                    and list(step.input_map.values())[0] == config_input.name
                    and step.model_name.startswith("snapshotter")
                ):
                    # only support streaming with batch size 1
                    shape[0] = 1

                    # now read the model config for the snapshotter to
                    # figure out what the names of its outputs are
                    snapshotter_config = self.client.get_model_config(
                        step.model_name
                    ).config

                    # iterate through the outputs of the snapshotter
                    # and figure out how many channels each of its
                    # states need to have. Record them in a dict
                    # mapping from the name of the state to the
                    # number of channels in the state
                    channel_map = {}
                    for x in snapshotter_config.output:
                        map_key = step.output_map[x.name]

                        # look for the model whose input
                        # gets fed by this output
                        for s in config.ensemble_scheduling.step:
                            for key, val in s.input_map.items():
                                if val == map_key:
                                    channel_name = f"{s.model_name}/{key}"
                                    shape = list(metadata_input.shape)
                                    shape[1] = x.dims[1]
                                    channel_map[channel_name] = tuple(shape)
                                    break
                            else:
                                continue

                    # add this state to our states
                    states.append((input, channel_map))
                    break
            else:
                # the loop didn't break, which means
                # that this input didn't meet the criterion
                # for being a snapshotter input, so just
                # add it as a regular input
                inputs.append(input)
        return inputs, states

    def run(self):
        with self.client:
            if len(self.states) > 0:
                self.client.start_stream(callback=self.callback)
            super().run()

    def cleanup(self, exc):
        # if we've run into an error, stop sending
        # requests to the server. Clear the client's
        # internal request queue and add its sentinel,
        # None, to indicate to kill its thread
        if len(self.states) > 0:
            self.client._stream._request_queue.queue.clear()
            self.client._stream._request_queue.put(None)

        # now do regular cleanup
        super().cleanup(exc)

    def _validate_package(
        self,
        package: Package,
        sequence_id: Optional[int],
        request_id: Optional[int],
        t0: Optional[int],
    ) -> Tuple[int, Optional[int], float]:
        # make sure that all of the timestamps corresponding
        # to the origin point of all packages are the same
        # TODO: is this a constraint we want to enforce?
        if t0 is None:
            t0 = package.t0
        elif t0 != package.t0:
            _raise_no_match("timestamps", t0, package.t0)

        # make sure that if one of the packages has
        # identified a sequence id thus far, that
        # all of the packages have that sequence id
        if sequence_id is None and package.sequence_id is not None:
            sequence_id = package.sequence_id
        elif (
            package.sequence_id is not None
            and sequence_id != package.sequence_id
        ):
            _raise_no_match("sequence id", sequence_id, package.sequence_id)

        # make sure that if one of the packages has
        # identified a request id thus far, that
        # all of the packages have that request id
        if request_id is None and package.request_id is not None:
            request_id = package.request_id
        elif request_id is None and package.request_id is None:
            # if there hasn't been a request id specified
            # yet and this package doesn't specify one, assign
            # a random one. Technically this could cause a
            # a collision, but what are the chances right?
            request_id = random.randint(0, 1e16)
        elif (
            package.request_id is not None and request_id != package.request_id
        ):
            _raise_no_match("request id", request_id, package.request_id)

        return sequence_id, request_id, t0

    def _validate_state(
        self,
        package: Package,
        name: str,
        shape: Tuple[int, ...],
        sequence_start: Optional[bool],
        sequence_end: Optional[bool],
    ) -> Tuple[bool, bool]:
        if package.x.shape != shape:
            raise ValueError(
                f"State {name} has shape {package.x.shape}, "
                f"but expected shape {shape}"
            )

        if sequence_start is None and package.sequence_start is not None:
            sequence_start = package.sequence_start
        elif (
            package.sequence_start is not None
            and package.sequence_start != sequence_start
        ):
            _raise_no_match(
                "sequence start flag", sequence_start, package.sequence_start
            )

        if sequence_end is None and package.sequence_end is not None:
            sequence_end = package.sequence_end
        elif (
            package.sequence_end is not None
            and package.sequence_end != sequence_end
        ):
            _raise_no_match(
                "sequence end flag", sequence_end, package.sequence_end
            )

        return sequence_start, sequence_end

    def get_package(self):
        packages = super().get_package()

        # if we have any non-stateful inputs, set their
        # input value using the corresponding package
        sequence_id, request_id, t0 = None, None, None
        for input in self.inputs:
            name = input.name()
            try:
                package = packages[name]
            except KeyError:
                raise ValueError(f"Missing input {name}")

            # make sure that any sequence id, request id,
            # or timestamps set on the package agree with
            # any values that we've seen so far
            sequence_id, request_id, t0 = self._validate_package(
                package, sequence_id, request_id, t0
            )
            input.set_data_from_numpy(package.x)

        # for any streaming input states, collect all
        # the updates for each state and set the input
        # message value using those updates. Do checks
        # on the sequence start and end values
        sequence_start, sequence_end = None, None
        for state, channel_map in self.states:
            states = []

            # for each update in the state, try to
            # get the update for it and do the checks
            for name, shape in channel_map.items():
                try:
                    package = packages[name]
                except KeyError:
                    raise ValueError(f"Missing state {name}")

                # validate the state update matches all
                # our expectations
                sequence_id, request_id, t0 = self._validate_package(
                    package, sequence_id, request_id, t0
                )
                sequence_start, sequence_end = self._validate_state(
                    package, name, shape, sequence_start, sequence_end
                )

                # add the update to our running list of updates
                states.append(package.x)

            # if we have more than one state, combine them
            # into a single tensor along the channel axis
            if len(states) > 1:
                states = [np.concatenate(states, axis=1)]

            # set the state input tensor with the
            # collected states if there are any
            if len(states) > 0:
                state.set_data_from_numpy(states[0])

        # record the start time of the message internally
        # for passing to downstream processes
        self.message_start_times[f"{request_id}_{sequence_id}"] = t0

        # return all the info we need to make
        # the appropriate inference call using
        # the newly set message values
        return request_id, sequence_id, sequence_start, sequence_end

    def clock_start(self, request_id: int, sequence_id: int) -> str:
        """Create a request id and record the start time for it"""

        # add on the sequence id to the request id if there
        # is a sequence associated with this request
        if sequence_id is not None:
            request_id = f"{request_id}_{sequence_id}"
        else:
            request_id = str(request_id)

        # if we're in profile mode, record the time at
        # which this request is "made" (really records
        # the time at which the request is put into Triton's
        # internal request queue, and also doesn't account
        # for message serialization time which is nontrivial)
        if self.profile:
            self._start_times[request_id] = time.time()

        # return the formatted request id
        # to associate with the request
        return request_id

    def clock_stop(self, request_id: str) -> Tuple[int, Optional[int], float]:
        """Get the start time for the request with the given id"""

        # if we're profiling, record the latency from when
        # the request was made to now and put it in the
        # profile queue to record round-trip latency
        if self.profile:
            start_time = self._start_times.pop(request_id)
            self._profile_q.put(time.time() - start_time)

        # get the time at which the _package_ was created
        # to set on the output package which gets placed
        # in the output queue
        t0 = self.message_start_times.pop(request_id)
        try:
            # see if there is a sequence associated
            # with this request and try to get its id
            request_id, sequence_id = map(int, request_id.split("_"))
        except ValueError:
            # if not, return None for the sequence id
            sequence_id = None

        return request_id, sequence_id, t0

    def callback(self, result, error=None) -> None:
        """Callback for Triton async inference calls"""

        # wrap everything in a try catch in case either
        # error is not `None` or something else goes
        # wrong so we can report the issue downstream.
        # Need to do this manually since this will run
        # in a thread and won't be caught by the try
        # catch in `self.run`
        try:
            # raise the error if anything went wrong
            if error is not None:
                # need to wrap Triton server exceptions since
                # they require an extra arg which throws off
                # the pickler. TODO: can we fix this in the
                # ExceptionWrapper class?
                # if isinstance(error, triton.InferenceServerException):
                error = RuntimeError(str(error))
                raise error

            # read the request id from the server response
            request_id = result.get_response().id

            # get the time associated with the generation of
            # the message, as well as the sequence id if there
            # is a sequence associated with the request
            request_id, sequence_id, t0 = self.clock_stop(request_id)

            # parse the numpy arrays from the response and
            # package them into a dictionary mapping from
            # the name of the output to the array
            np_output = {}
            for output in result._result.outputs:
                np_output[output.name] = Package(
                    x=result.as_numpy(output.name),
                    t0=t0,
                    request_id=request_id,
                    sequence_id=sequence_id,  # TODO: non-states don't need
                )

            # send these parsed outputs to downstream processes
            self.out_q.put(np_output)
        except Exception as e:
            # since this is executing asynchronously, wait until
            # the logger gets initialized before doing cleanup
            while self.logger is None:
                time.sleep(1e-6)

            # run cleanup, which should stop the process
            self.cleanup(e)

    def process(self, request_id, sequence_id, sequence_start, sequence_end):
        request_id = self.clock_start(request_id, sequence_id)
        if sequence_id is not None:
            self.client.async_stream_infer(
                self.model_name,
                model_version=str(self.model_version),
                inputs=self.inputs + [x[0] for x in self.states],
                request_id=request_id,
                sequence_id=sequence_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                timeout=60,
            )
        else:
            self.client.async_infer(
                self.model_name,
                model_version=str(self.model_version),
                inputs=self.inputs,
                request_id=request_id,
                callback=self.callback,
                timeout=60,
            )
