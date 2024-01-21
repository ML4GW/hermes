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
from copy import deepcopy
from queue import Empty, Queue
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

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
    """
    Check if the given model and version are ready on an inference
    service instance which is being connected to by `client`
    """

    if not client.is_model_ready(model_name, model_version):
        # the desired model and version aren't ready, so
        # see if we can load the model into the service
        try:
            logging.info(f"Attempting to load model {model_name}")
            client.load_model(model_name)
            logging.info(f"Model {model_name} loaded")
        except InferenceServerException as e:
            # catch an exeption indicating that Triton doesn't
            # have explicit control mode on, and raise it
            # more explicitly
            if str(e).endswith("polling is enabled"):
                raise RuntimeError(
                    "Model {}, version {} isn't ready on server "
                    "and explicit model control not enabled".format(
                        model_name, model_version
                    )
                )
            else:
                raise

    if not client.is_model_ready(model_name, model_version):
        # evidently we were able to load the model, but
        # the desired version didn't get loaded with it,
        # so we're out of options here and need to bail
        raise RuntimeError(
            "Model {} is available but not version {}".format(
                model_name, model_version
            )
        )


class InferenceClient:
    """Connect to a Triton server instance and make requests to it

    Connect to a Triton inference service instance running at
    the specified address for performing inference with the
    indicated model. Dynamically infer the names, shapes, and
    datatypes of the inputs to the model, as well as whether
    any of those inputs are meant to be stateful.

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
            this will need to be padded since the full batch
            dimension will be strictly enforced.
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
        self.model_name = model_name
        self.inputs, self.states = self._build_inputs(batch_size)
        self.num_states = sum([len(i[1]) for i in self.states])

        # infer the model version and make sure that the desired
        # model is ready to have requests made to it
        if model_version == -1:
            model_version = max(map(int, self.metadata.versions))
        _check_ready(self.client, model_name, str(model_version))
        self.model_version = model_version

        # set some things up for the streaming callback
        self.callback = callback
        if profile:
            self.clock = ProfilerClock()
        else:
            self.clock = None
        self.callback_q = Queue()
        self._sequences = {}

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

        # TODO: check for stateful outputs for enforcing a
        # sequence_id value in `InferenceClient.infer`
        states, inputs = [], []
        for config_input, metadata_input in zip(config.input, metadata.inputs):
            shape = [i if i > 0 else batch_size for i in metadata_input.shape]
            input = triton.InferInput(
                name=metadata_input.name,
                shape=shape,
                datatype=metadata_input.datatype,
            )
            for step in config.ensemble_scheduling.step:
                # see if this input corresponds to
                # the input for a snapshotter model
                input_map = list(step.input_map.values())
                if len(input_map) == 1 and input_map[0] == config_input.name:
                    model_config = self.client.get_model_config(
                        step.model_name
                    )
                    model_config = model_config.config
                    if len(model_config.sequence_batching.state) == 0:
                        continue

                    # only support streaming with batch size 1
                    shape[0] = 1

                    # iterate through the outputs of the snapshotter
                    # and figure out how many channels each of its
                    # states need to have. Record them in a dict
                    # mapping from the name of the state to the
                    # number of channels in the state
                    channel_map = {}
                    for x in model_config.output:
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

    def get(self, until_empty: bool = False) -> Union[Any, List[Any], None]:
        """Check the if the callback thread has produced anything

        Makes a check to see if the callback thread has produced
        any inference responses since the last check. If any
        errors have been raised in the callback, they will get
        raised with their traceback here.

        Args:
            until_empty:
                If `True`, grab server responses from the
                `callback_q` until `queue.Empty` gets raised,
                and return a list containing all the responses.
                Otherwise, return the first reponse produced by
                the `callback_q`.
        Returns:
            If `until_empty` is `True`, returns a `list` of
            each of the server responses waiting in the
            `callback_q`. Otherwise, if there is a response
            in the `callback_q`, return it, returning `None`
            if the queue is empty.
        Raises:
            Exception:
                If `sys.exc_info()` is retrieved from the
                `callback_q`, the exception will be raised
                with its traceback in the callback thread.
        """

        responses = []
        while True:
            try:
                response = self.callback_q.get_nowait()
            except Empty:
                # there's nothing in the queue
                if until_empty:
                    # stop looking for more responses and
                    # return whatever we found, even if
                    # its an empty list
                    break

                # otherwise return `None`
                return
            else:
                if (
                    isinstance(response, tuple)
                    and len(response) == 3
                    and isinstance(response[1], Exception)
                ):
                    # complicated check that this looks like the
                    # return from sys.exc_info(), indicating that
                    # something went wrong in the callback
                    _, exc, tb = response
                    raise exc.with_traceback(tb)

                if not until_empty:
                    # we got a response and we only asked
                    # for one, so return it
                    return response
                else:
                    # otherwise add it to our running list
                    responses.append(response)

        # this must mean `until_empty is True`, so
        # return all the responses we got, even if
        # there were no responses at all
        return responses

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

    def _validate_inputs(
        self,
        x: Union[np.ndarray, Dict[str, np.ndarray]],
        sequence_id: Optional[int] = None,
    ):
        """
        Normalize an inference input and grab the gRPC
        InferenceInput objects it will we used to fill
        with data in a thread-safe fashion.
        """
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
                # if we only have a non-stateful input, set `x`
                # up to keep the parsing methods below more standard
                x = {self.inputs[0].name(): x}
            else:
                # same for if we only have a stateful input
                state_name = list(self.states[0][1].keys())[0]
                x = {state_name: x}

        # now do some checks on the sequence that this input
        # belongs to (if any) and grab or create the corresponding
        # InferenceInputs it will be used to fill with data
        if sequence_id is None and len(self.states) > 0:
            # enforce that we provide a sequence id if
            # there are any stateful inputs. TODO: should
            # we do the same if there are any stateful outputs?
            # Or just states somewhere in the model in general?
            raise ValueError(
                "Must provide sequence id for model with states {}".format(
                    [i[0].name() for i in self.states]
                )
            )
        elif sequence_id is not None and self.num_states == 0:
            raise ValueError(
                "Specified sequence id {} for request to "
                "non-stateful model {}".format(sequence_id, self.model_name)
            )
        elif sequence_id is not None and sequence_id not in self._sequences:
            # this is a new sequence, so create a fresh set of inputs for it
            # to make doing inference across multiple streams thread-safe
            logging.debug(
                f"Creating new inputs and states for sequence {sequence_id}"
            )
            inputs, states = deepcopy(self.inputs), deepcopy(self.states)
            self._sequences[sequence_id] = (inputs, states)
        elif sequence_id is not None:
            # otherwise this is an existing sequence, so grab
            # the corresponding inputs and states
            inputs, states = self._sequences[sequence_id]
        elif self.num_states == 0:
            # we're not doing stateful inference, so there's
            # no sequences to keep track of in the first place
            inputs, states = self.inputs, []
        return x, inputs, states, sequence_id

    def infer(
        self,
        x: Union[np.ndarray, Dict[str, np.ndarray]],
        request_id: Optional[int] = None,
        sequence_id: Optional[int] = None,
        sequence_start: bool = False,
        sequence_end: bool = False,
    ) -> None:
        """Make an asynchronous inference request to the service

        Use the indicated input or inputs to make an inference
        request to the model sitting on the inference service.
        If this model requires multiple inputs or states, `x`
        should be a dictionary mapping from the name of each
        input or state to the corresponding value. Otherwise,
        `x` may be a single numpy array representing the input
        data for this inference request.

        Responses from the inference service will be handled
        in an asynchronous callback thread, with the parsed and
        postprocessed values placed in this object's `callback_q`.
        As a simple way to retrieve the values from that queue,
        consider using the `InferenceClient.get` method.

        Responses follow the same structure as the inputs: if
        there are multiple outputs, the response will be a
        dictionary mapping from output names to values. If there
        is only a single output, it will be returned as a NumPy
        array.

        Args:
            x:
                The inputs to the model sitting on the server.
                If the model has multiple inputs (stateful or
                stateless), this should be a `dict` mapping from
                input names to corresopnding NumPy arrays. If
                the model has only a single input, this may be
                a single NumPy array containing that input's value.
            request_id:
                An identifier to associate with this inference request.
                Will be passed along to the `callback` specified
                at initialization, or if this is `None` it will
                be placed in the `callback_q` alongside the response
                values.
            sequence_id:
                An identifier to associate this request with a particular
                state on the inference server. Required if the
                indicated model has any stateful inputs, otherwise
                won't do anything.
            sequence_start:
                Indicates whether this request is the first in a
                new sequence. Won't do anything if the
                model has no stateful inputs.
            sequence_end:
                Indicates whether this request is the final one
                in a sequence. Won't do anything if the model has
                no stateful inputs.
        """

        x, inputs, states, sequence_id = self._validate_inputs(x, sequence_id)

        # if we have any non-stateful inputs, set their
        # input value using the corresponding package
        for input in inputs:
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
        for state, channel_map in states:
            state_values = []

            # for each update in the state, try to
            # get the update for it and do the checks
            for name, shape in channel_map.items():
                try:
                    value = x[name]
                except KeyError:
                    raise ValueError(f"Missing state {name}")

                # add the update to our running list of updates
                state_values.append(value[None])

            # if we have more than one state, combine them
            # into a single tensor along the channel axis
            if len(state_values) > 1:
                state = np.concatenate(state_values, axis=1)
                state.set_data_from_numpy(state)
            else:
                state.set_data_from_numpy(state_values[0])

        # if a request_id wasn't specified, give it a random
        # one that will (probably) be unique
        if request_id is None:
            request_id = random.randint(0, 1e16)

        # keep track of in-flight times if we're profiling
        if self.clock is not None:
            self.clock.tick(request_id, sequence_id)

        # the server response won't contain info about
        # the sequence id, so if we have one attach it
        # to the request id so we can parse in the callback
        if sequence_id is not None:
            request_id = f"{request_id}_{sequence_id}"
        else:
            request_id = str(request_id)

        if len(self.states) > 0:
            # make a streaming inference if we have input states
            # TODO: don't restrict ourselves just to input states
            # but having stateful behavior anywhere in the model
            self.client.async_stream_infer(
                self.model_name,
                model_version=str(self.model_version),
                inputs=inputs + [x[0] for x in states],
                request_id=request_id,
                sequence_id=sequence_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                timeout=60,
            )

            if sequence_end:
                # remove the inputs for this sequence if it's complete
                self._sequences.pop(sequence_id)
        else:
            self.client.async_infer(
                self.model_name,
                model_version=str(self.model_version),
                inputs=self.inputs,
                request_id=request_id,
                callback=self._callback,
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

            # if there's only one output, just return the numpy
            # array and spare the user the need to keep track
            # of input and output names
            if len(np_output) == 1:
                np_output = np_output[output.name]

            # send these parsed outputs to downstream processes
            response = (np_output, request_id, sequence_id)
            if self.callback is not None:
                response = self.callback(*response)

            # give callbacks the option of returning `None`
            # if all they have is some intermediate product
            # that they don't want shipped back to the main thread.
            if response is not None:
                self.callback_q.put(response)

        except Exception:
            self.callback_q.put(sys.exc_info())
