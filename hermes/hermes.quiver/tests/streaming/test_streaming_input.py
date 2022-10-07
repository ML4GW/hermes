from unittest.mock import MagicMock

import numpy as np
import pytest

from hermes.quiver import ModelRepository


@pytest.fixture(params=[1, 4, 100])
def snapshot_size(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def stride_size(request):
    return request.param


@pytest.fixture(params=[[1], [4], [1, 4], [1, 2, 4], [1, 0]])
def channels(request):
    return request.param


@pytest.fixture
def snapshotter(snapshot_size, stride_size, batch_size, channels):
    from hermes.quiver.streaming.streaming_input import Snapshotter

    return Snapshotter(snapshot_size, stride_size, batch_size, channels)


@pytest.mark.torch
def test_snapshotter(snapshot_size, stride_size, batch_size, channels):
    import torch

    from hermes.quiver.streaming.streaming_input import Snapshotter

    snapshotter = Snapshotter(snapshot_size, stride_size, batch_size, channels)
    num_channels = sum([i or 1 for i in channels])

    # now run an input through as a new sequence and
    # make sure we get the appropriate number of outputs
    # if our update size is too large, this should raise
    # an error then we're done testing this combination
    update_size = stride_size * batch_size
    if update_size > snapshot_size:
        return

    snapshot = torch.arange((snapshot_size + update_size) * num_channels)
    snapshot = snapshot.reshape(1, num_channels, snapshot_size + update_size)
    snapshot, update = torch.split(
        snapshot, [snapshot_size, update_size], dim=-1
    )
    snapshot = snapshot.type(torch.float32)
    update = update.type(torch.float32)

    outputs = snapshotter(update, snapshot)
    outputs = [i.cpu().numpy() for i in outputs]
    new_snapshot = outputs.pop(-1)

    assert len(outputs) == len(channels)
    offset = stride_size
    for k, (output, channel_dim) in enumerate(zip(outputs, channels)):
        expected = (i for i in (batch_size, channel_dim, snapshot_size) if i)
        assert output.shape == tuple(expected)
        if channel_dim == 0:
            output = output[:, None]

        for i, row in enumerate(output):
            for j, channel in enumerate(row):
                start = j * (snapshot_size + update_size) + i * stride_size
                stop = start + snapshot_size
                expected = np.arange(start, stop) + offset
                assert (channel == expected).all(), (k, i, j)
        offset += channel_dim * (snapshot_size + update_size)

    assert new_snapshot.shape == (1, num_channels, snapshot_size)
    for i, channel in enumerate(new_snapshot[0]):
        start = update_size + i * (snapshot_size + update_size)
        stop = start + snapshot_size
        expected = np.arange(start, stop)
        assert (channel == expected).all()


@pytest.fixture(scope="function")
def very_temp_local_repo():
    repo = ModelRepository("hermes-quiver-test-transient")
    yield repo
    repo.delete()


@pytest.mark.torch
@pytest.mark.parametrize("streams_per_gpu", [1, 2])
def test_make_streaming_input_model(
    very_temp_local_repo,
    snapshot_size,
    stride_size,
    batch_size,
    channels,
    streams_per_gpu,
):
    from hermes.quiver.streaming.streaming_input import (
        make_streaming_input_model,
    )

    inputs = []
    names = "abcdefg"[: len(channels)]
    for channel, name in zip(channels, names):
        x = MagicMock()
        x.name = name
        x.model.name = "my-model"
        if channel == 0:
            x.shape = (batch_size, snapshot_size)
        else:
            x.shape = (batch_size, channel, snapshot_size)
        inputs.append(x)

    update_size = stride_size * batch_size
    if update_size > snapshot_size:
        with pytest.raises(ValueError):
            model = make_streaming_input_model(
                very_temp_local_repo,
                inputs,
                stride_size,
                batch_size,
                streams_per_gpu=streams_per_gpu,
            )
        return

    model = make_streaming_input_model(
        very_temp_local_repo,
        inputs,
        stride_size,
        batch_size,
        streams_per_gpu=streams_per_gpu,
    )
    assert model.name == "snapshotter"

    config_path = very_temp_local_repo.fs.join("snapshotter", "config.pbtxt")
    config = very_temp_local_repo.fs.read_config(config_path)

    num_channels = sum([i or 1 for i in channels])
    assert len(config.input) == 1
    assert config.input[0].name == "snapshot_update"
    assert config.input[0].dims == [1, num_channels, update_size]

    assert len(config.output) == len(channels)
    for channel, name, output in zip(channels, names, config.output):
        assert output.name == f"my-model.{name}_snapshot"

        expected_shape = [i for i in [batch_size, channel, snapshot_size] if i]
        assert output.dims == expected_shape

    assert len(config.sequence_batching.state) == 1
    assert config.sequence_batching.state[0].input_name == "input_snapshot"
    assert config.sequence_batching.state[0].output_name == "output_snapshot"
    assert config.sequence_batching.state[0].dims == [
        1,
        num_channels,
        snapshot_size,
    ]

    assert len(config.instance_group) == 1
    assert config.instance_group[0].count == streams_per_gpu
