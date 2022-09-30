from unittest.mock import MagicMock

import pytest


@pytest.fixture(params=[1, 2, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def update_size(request):
    return request.param


@pytest.fixture(params=[1, 10])
def num_updates(request):
    return request.param


@pytest.mark.torch
@pytest.fixture
def averager(batch_size, update_size, num_updates):
    from hermes.quiver.streaming.streaming_output import OnlineAverager

    return OnlineAverager(
        update_size=update_size, batch_size=batch_size, num_updates=num_updates
    )


@pytest.mark.torch
def test_online_averager(averager):
    import torch

    num_updates = averager.num_updates
    update_size = averager.update_size
    batch_size = averager.batch_size

    # make a batch of overlapping aranged data such that
    # the online average is just the values themselves
    size = 2 * batch_size * (update_size - 1) + 200
    x = torch.arange(size).view(1, 1, 1, -1).type(torch.float32)
    x = torch.nn.functional.unfold(
        x, (1, 2 * batch_size), dilation=(1, update_size)
    )[0]

    # initialize a blank initial snapshot
    snapshot_size = update_size * (batch_size + num_updates)
    snapshot = torch.zeros((snapshot_size,))

    # initialize an update index
    update_idx = torch.zeros((1,))

    # perform the first aggregation step
    stream, new_snapshot, update_idx = averager(
        x[:batch_size], snapshot, update_idx
    )

    # make sure the shapes are right
    assert stream.shape == (1, update_size * batch_size)
    assert new_snapshot.shape == snapshot.shape
    assert update_idx.item() == batch_size

    # now validate that the streamed value is correct
    start = size - update_size * (num_updates + batch_size - 1)
    start -= update_size * batch_size
    stop = start + update_size * batch_size
    expected = torch.arange(start, stop)
    assert (stream[0] == expected).all().item()

    # finally check that the snapshot values are as expected
    filled = (num_updates - 1) * update_size
    expected = torch.arange(stop, stop + filled)
    assert (new_snapshot[:filled] == expected).all().item()
    assert (new_snapshot[filled:] == 0).all().item()

    # now take the next step and confirm everything again
    stream, newer_snapshot, update_idx = averager(
        x[batch_size:], new_snapshot, update_idx
    )
    assert stream.shape == (1, update_size * batch_size)
    assert new_snapshot.shape == snapshot.shape
    assert update_idx.item() == 2 * batch_size

    start = size - update_size * (num_updates + batch_size - 1)
    stop = start + update_size * batch_size
    expected = torch.arange(start, stop)
    assert (stream[0] == expected).all().item()

    expected = torch.arange(stop, stop + filled)
    assert (newer_snapshot[:filled] == expected).all().item()
    assert (newer_snapshot[filled:] == 0).all().item()


@pytest.mark.torch
def test_make_streaming_output_model(
    update_size, batch_size, num_updates, temp_local_repo
):
    from hermes.quiver.streaming.streaming_output import (
        make_streaming_output_model,
    )

    input = MagicMock()
    input.shape = (batch_size, 128)

    model = make_streaming_output_model(
        temp_local_repo, input, update_size, batch_size, num_updates
    )

    assert len(model.config.input) == 1
    assert model.config.input[0].name == "update"
    assert model.config.input[0].dims == [batch_size, 128]

    assert len(model.config.output) == 1
    assert model.config.output[0].name == "stream"
    assert model.config.output[0].dims == [1, update_size * batch_size]

    assert len(model.config.sequence_batching.state) == 2
    assert model.config.sequence_batching.state[0].input_name == (
        "input_online_average"
    )
    assert model.config.sequence_batching.state[0].output_name == (
        "output_online_average"
    )
    assert model.config.sequence_batching.state[0].dims == [
        update_size * (batch_size + num_updates)
    ]

    assert model.config.sequence_batching.state[1].input_name == (
        "input_update_index"
    )
    assert model.config.sequence_batching.state[1].output_name == (
        "output_update_index"
    )
    assert model.config.sequence_batching.state[1].dims == [1]
