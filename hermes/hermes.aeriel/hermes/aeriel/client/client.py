# TODO: create custom future object which inherits from
# concurrent.futures.Future and store in an
# `InferenceClient.futures` dictionary that maps from a
# (request_id, sequence_id) to the corresponding Future,
# use request_id and sequence_id in callback to pop the
# corresponding future and use `Future.set_result` or
# `Future.set_exception` as required. Set request_id
# and sequence_id attributes on the Future for use by
# calling processes.

import logging
import random
import sys
import time
from collections import defaultdict
from queue import Empty, Queue
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tritonclient.grpc as triton
from tritonclient.utils import InferenceServerException

if TYPE_CHECKING:
    from tritonclient.grpc.model_config_pb2 import InferInput


SHAPE = Tuple[int, ...]


class ProfilerClock:
    def __init__(self):
        self._start_times = {}
        self.latencies = defaultdict(list)

    def tick(self, request_id, sequence_id):
        self._start_times[(request_id, sequence_id)] = time.time()

    def tock(self, request_id, sequence_id):
        end_time = time.time()
        start_time = self._start_times.pop((request_id, sequence_id))
        self.latencies[sequence_id].append(end_time - start_time)


def _check_ready(
    client: triton.InferenceServerClient, model_name: str, model_version: str
) -> None:
    if not client.is_model_ready(model_name, model_version):
        try:
            logging.info(f"Attempting to load model {model_name}")
            client.load_model(model_name)
            logging.info(f"Model {model_name} loaded")
        except InferenceServerException as e:
            if str(e).endswith("polling is enabled"):
                raise RuntimeError(
                    "Model {}, version {} isn't ready on server "
                    "and explicit model control not enabled".format(
                        model_name, model_version
                    )
                )
            else:
                raise
        else:
            raise RuntimeError(
                "Model {} version {} isn't ready on server".format(
                    model_name, model_version
                )
            )
    if not client.is_model_ready(model_name, model_version):
        raise RuntimeError(
            "Model {} is available but not version {}".format(
                model_name, model_version
            )
        )


class InferenceClient:
    """Process for making asynchronous requests to a Triton inference service

    Make asynchronous requests to a Triton inference service
    in a separate process in order to make requests in tandem
    with data pre- and post-processing. Uses model metadata to
    build input protobufs and dynamically detects the presence
    of snapshotter states, exposing metadata about those states
    for building data generators.

    Args:
        address:
            The url at which the service is being hosted
        model_name:
            The name of the model to which to make requests
        model_version:
            The version of the model to which to make requests. If
            set to `-1`, the latest version of the model will be
            dynamically inferred.
        batch_size:
            The batch size to use for the non-stateful inputs
            to the model. Inputs with batch sizes smaller than
            this will need to be padded.
        callback:
            Optional function to call on the parsed response
            from the inference service
        profile:
            Whether to record round-trip latencies to and from
            the server in a `self.clock` attribute.
        client_kwargs:
            Additional kwargs to pass to the Triton
            `InferenceServerClient`
    """

    def __init__(
        self,
        address: str,
        model_name: str,
        model_version: int = -1,
        batch_size: int = 1,
        callback: Optional[Callable] = None,
        profile: bool = False,
        **client_kwargs,
    ) -> None:
        client = triton.InferenceServerClient(address, **client_kwargs)
        try:
            if not client.is_server_live():
                raise RuntimeError(f"Server at url {address} isn't live")
        except triton.InferenceServerException:
            raise RuntimeError(f"Couldn't connect to server at {address}")
        self.client = client
        self.address = address

        # `inputs` will be a list of Triton `InferInput`
        # objects representing the non-stateful inputs
        # to the model
        # `states` will be a list of tuples where the first
        # input will be a Triton `InferInput` object representing
        # an input to a snapshotter model, and the second
        # will be a dictionary mapping from downstream input names
        # that the snapshotter feeds to the shape of that input
        self.inputs, self.states = self._build_inputs(batch_size)
        self.num_states = sum([len(i[1]) for i in self.states])

        # infer the model version and make sure that the desired
        # model is ready to have requests made to it
        if model_version == -1:
            model_version = max(map(int, self.metadata.versions))
        _check_ready(self.client, model_name, str(model_version))

        self.model_name = model_name
        self.model_version = model_version

        # set some things up for the streaming callback
        self.callback = callback
        if profile:
            self.clock = ProfilerClock()
        else:
            self.clock = None
        self.callback_q = Queue()

    def _build_inputs(
        self, batch_size: int
    ) -> Tuple[
        List["InferInput"],
        List[Tuple["InferInput", Dict[str, SHAPE]]],
    ]:
        """Build Triton InferInputs for the inputs to the model

        Use the metadata returned by the inference server about
        the model to infer the names, shapes, and datatypes of
        model inputs and build the corresponding protobuf objects
        using this info.

        For models with streaming input states, infer the different
        input states associated with a snapshot and the order in
        which they're concatenated along the channel axis.
        """

        # TODO: confirm that batch_size is not larger
        # than config.max_batch_size?
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

        # record the config and metadata for external reference
        self.config = config
        self.metadata = metadata

        return inputs, states

    def get(self):
        """Check if the callback thread has run into an error"""
        try:
            response = self.callback_q.get_nowait()
        except Empty:
            return
        else:
            if isinstance(response[0], Exception):
                _, exc, tb = response
                raise exc.with_traceback(tb)
            return response

    def __enter__(self):
        self.client.__enter__()
        if len(self.states) > 0:
            self.client.start_stream(callback=self._callback)
        return self

    def __exit__(self, *exc_args):
        # if we've run into an error, stop sending
        # requests to the server. Clear the client's
        # internal request queue and add its sentinel,
        # None, to indicate to kill its thread
        if len(self.states) > 0 and exc_args[0] is not None:
            self.client._stream._request_queue.queue.clear()
            self.client._stream._request_queue.put(None)
        self.client.__exit__(*exc_args)

    def infer(
        self,
        x: Union[np.ndarray, Dict[str, np.ndarray]],
        request_id: Optional[int] = None,
        sequence_id: Optional[int] = None,
        sequence_start: bool = False,
        sequence_end: bool = False,
    ):
        # if we just passed a single array, make sure we only
        # have one input or state that we need to pass it to
        if not isinstance(x, dict):
            if len(self.inputs) + self.num_states > 1:
                raise ValueError(
                    "Only passed a single input array, but "
                    "model {} has {} inputs and {} states".format(
                        self.model_name, len(self.inputs), len(self.states)
                    )
                )
            elif len(self.inputs) > 0:
                x = {self.inputs[0].name(): x}
            else:
                state_name = list(self.states[0][1].keys())[0]
                x = {state_name: x}

        if request_id is None:
            request_id = random.randint(0, 1e16)
        if sequence_id is None and len(self.states) > 0:
            raise ValueError(
                "Must provide sequence id for model with states {}".format(
                    [i[0].name() for i in self.states]
                )
            )

        # if we have any non-stateful inputs, set their
        # input value using the corresponding package
        for input in self.inputs:
            name = input.name()
            try:
                value = x[name]
            except KeyError:
                raise ValueError(f"Missing input {name}")

            # TODO: Should pad inputs with batches smaller than
            # batch size, and record it somewhere for
            # slicing in the callback? Also raise an error
            # for batches that are larger than batch size?
            input.set_data_from_numpy(value)

        # for any streaming input states, collect all
        # the updates for each state and set the input
        # message value using those updates. Do checks
        # on the sequence start and end values
        for state, channel_map in self.states:
            states = []

            # for each update in the state, try to
            # get the update for it and do the checks
            for name, shape in channel_map.items():
                try:
                    value = x[name]
                except KeyError:
                    raise ValueError(f"Missing state {name}")

                # add the update to our running list of updates
                states.append(value[None])

            # if we have more than one state, combine them
            # into a single tensor along the channel axis
            if len(states) > 1:
                state = np.concatenate(states, axis=1)
                state.set_data_from_numpy(state)
            else:
                state.set_data_from_numpy(states[0])

        # keep track of in-flight times if we're profiling
        if self.clock is not None:
            self.clock.tick(request_id, sequence_id)

        # the server response won't contain info about
        # the sequence id, so if we have one attach it
        # to the request id so we can parse in the callback
        if sequence_id is not None:
            request_id = f"{request_id}_{sequence_id}"

        if len(self.states) > 0:
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

    def _callback(self, result, error=None) -> None:
        """Callback for Triton async inference calls"""

        # wrap everything in a try catch in case either
        # error is not `None` or something else goes
        # wrong so we can report the issue downstream.
        try:
            # raise the error if anything went wrong
            if error is not None:
                error = RuntimeError(str(error))
                raise error

            # read the request id from the server response
            request_id = result.get_response().id

            try:
                # see if there is a sequence associated
                # with this request and try to get its id
                request_id, sequence_id = map(int, request_id.split("_"))
            except ValueError:
                # if not, return None for the sequence id
                request_id = int(request_id)
                sequence_id = None

            # if we're profiling, record the in-flight latency
            # associated with this request
            if self.clock is not None:
                self.clock.tock(request_id, sequence_id)

            logging.debug(
                "Received response for request {} from sequence {}".format(
                    request_id, sequence_id
                )
            )

            # parse the numpy arrays from the response and
            # package them into a dictionary mapping from
            # the name of the output to the array
            np_output = {}
            for output in result._result.outputs:
                np_output[output.name] = result.as_numpy(output.name)

            if len(np_output) == 1:
                np_output = np_output[output.name]

            # send these parsed outputs to downstream processes
            response = (np_output, request_id, sequence_id)
            if self.callback is not None:
                response = self.callback(*response)

            if response is not None:
                self.callback_q.put(response)

        except Exception:
            self.callback_q.put(sys.exc_info())
