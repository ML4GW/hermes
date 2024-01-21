from dataclasses import dataclass
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hermes.aeriel.client import InferenceClient


@dataclass
class DummyInferInput:
    _name: str
    shape: Tuple[int, ...]

    def __post_init__(self):
        self.x = None

    def name(self):
        return self._name

    def set_data_from_numpy(self, x):
        self.x = x


class DummyMetadata:
    versions = ["1", "3"]


class AddOneStream:
    def __init__(self, callback=None):
        self.callback = callback

    def __call__(self, *args, **kwargs):
        response = MagicMock()
        response.id = kwargs["request_id"]

        result = MagicMock()
        result.get_response = MagicMock(return_value=response)

        output = MagicMock()
        output.name = "output"
        result._result.outputs = [output]

        answer = kwargs["inputs"][-1].x.reshape(-1)[-1] + 1
        result.as_numpy = MagicMock(return_value=answer)

        callback = self.callback or kwargs["callback"]
        callback(result, None)


@pytest.fixture(params=[0, 1, 2])
def num_inputs(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def num_states(request):
    return request.param


@pytest.fixture(params=[-1, 1, 2, 3])
def version(request):
    return request.param


# TODO: create control_mode param and create load_model patch
# inside of function, raising error if control_mode != "explicit"
@patch(
    "tritonclient.grpc.InferenceServerClient.is_server_live", return_value=True
)
@patch(
    "tritonclient.grpc.InferenceServerClient.is_model_ready",
    new=lambda obj, _, v: v == DummyMetadata.versions[-1],
)
@patch(
    "tritonclient.grpc.InferenceServerClient.load_model",
)
def test_inference_client(mock1, mock2, num_inputs, num_states, version):
    if num_inputs + num_states == 0:
        return

    # TODO: parametrize batch size?
    num_channels = 3
    dim = 128
    shape = (1, num_channels, dim)
    inputs = [DummyInferInput(f"input{i}", shape) for i in range(num_inputs)]

    if num_states > 0:
        channels = [i + 1 for i in range(num_states)]
        total = sum(channels)
        dim = 32
        states = {f"state{i}": (j, dim) for i, j in enumerate(channels)}
        states = [(DummyInferInput("state", (1, total, dim)), states)]
    else:
        states = []

    with patch(
        "hermes.aeriel.client.InferenceClient._build_inputs",
        return_value=(inputs, states),
    ):
        InferenceClient.metadata = DummyMetadata
        if version not in [-1, 3]:
            with pytest.raises(RuntimeError):
                client = InferenceClient(
                    "localhost:8001", "dummy-model", version
                )
            return

        postprocessor = MagicMock()
        client = InferenceClient(
            "localhost:8001",
            "dummy-model",
            version,
            callback=postprocessor,
        )

        method = "async_stream_infer" if num_states > 0 else "async_infer"
        new = AddOneStream(client._callback)
        with patch(
            f"tritonclient.grpc.InferenceServerClient.{method}", new=new
        ):
            if (num_inputs + num_states) == 1:
                if num_states == 1:
                    x = np.random.randn(*states[0][0].shape)
                    kwargs = {
                        "sequence_id": 1001,
                        "sequence_start": MagicMock(),
                    }
                else:
                    x = np.random.randn(1, num_channels, dim)
                    kwargs = {}

                client.infer(x, request_id=10, **kwargs)
                x_expected = x.reshape(-1)[-1] + 1
                if num_states > 0:
                    postprocessor.assert_called_with(x_expected, 10, 1001)
                else:
                    postprocessor.assert_called_with(x_expected, 10, None)
