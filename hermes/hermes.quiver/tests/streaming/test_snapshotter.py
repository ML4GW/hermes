import numpy as np
import pytest
import torch


@pytest.fixture(params=[1, 4, 100])
def snapshot_size(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def stride_size(request):
    return request.param


@pytest.fixture(params=[[1], [4], [1, 4], [1, 2, 4]])
def channels(request):
    return request.param


@pytest.fixture
def snapshotter(snapshot_size, stride_size, batch_size, channels):
    from hermes.quiver.streaming.streaming_input import Snapshotter

    return Snapshotter(snapshot_size, stride_size, batch_size, channels)


@pytest.mark.torch
def test_snapshotter(snapshot_size, stride_size, batch_size, channels):
    from hermes.quiver.streaming.streaming_input import Snapshotter

    snapshotter = Snapshotter(snapshot_size, stride_size, batch_size, channels)
    num_channels = sum(channels)

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
        assert output.shape == (batch_size, channel_dim, snapshot_size)
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
