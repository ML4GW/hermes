from unittest.mock import MagicMock

import pytest


@pytest.mark.torch
def test_online_averager_simple():
    import torch

    from hermes.quiver.streaming.streaming_output import OnlineAverager

    x = torch.arange(8).view(1, -1).repeat(4, 1).type(torch.float32) + 1
    snapshot = torch.zeros((11,))
    update_idx = torch.zeros((1,))

    averager = OnlineAverager(
        update_size=1, batch_size=4, num_updates=8, num_channels=None
    )

    y, snapshot, update_idx = averager(x, snapshot, update_idx)
    assert (y == torch.tensor([1, 1.5, 2, 2.5])).all().item()

    expected_snapshot = [3.5, 4.5, 5.5, 6.5, 7, 7.5, 8, 0, 0, 0, 0]
    assert (snapshot == torch.tensor(expected_snapshot)).all().item()
    assert update_idx.item() == 4

    y, snapshot, update_idx = averager(x, snapshot, update_idx)
    assert (y == torch.tensor([3, 3.5, 4, 4.5])).all().item()

    expected_snapshot = [5, 5.5, 6, 6.5, 7, 7.5, 8, 0, 0, 0, 0]
    assert (snapshot == torch.tensor(expected_snapshot)).all().item()
    assert update_idx.item() == 8

    y, snapshot, update_idx = averager(x, snapshot, update_idx)
    assert (y == 4.5).all().item()
    assert (snapshot == torch.tensor(expected_snapshot)).all().item()
    assert update_idx.item() == 12


@pytest.fixture(params=[1, 2, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def update_size(request):
    return request.param


@pytest.fixture(params=[1, 10])
def num_updates(request):
    return request.param


@pytest.fixture(params=[None, 2])
def num_channels(request):
    return request.param


@pytest.mark.torch
@pytest.fixture
def averager(batch_size, update_size, num_updates, num_channels):
    from hermes.quiver.streaming.streaming_output import OnlineAverager

    return OnlineAverager(
        update_size=update_size,
        batch_size=batch_size,
        num_updates=num_updates,
        num_channels=num_channels,
    )


@pytest.mark.torch
def test_online_averager(averager, num_channels):
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

    if num_channels is not None:
        x = x.view(2 * batch_size, 1, -1)
        x = torch.repeat_interleave(x, num_channels, axis=1)

    # initialize a blank initial snapshot
    snapshot_size = update_size * (batch_size + num_updates - 1)
    snapshot_shape = (snapshot_size,)
    if num_channels is not None:
        snapshot_shape = (num_channels,) + snapshot_shape
    snapshot = torch.zeros(snapshot_shape)

    # initialize an update index
    update_idx = torch.zeros((1,))

    # perform the first aggregation step
    stream, new_snapshot, update_idx = averager(
        x[:batch_size], snapshot, update_idx
    )

    # make sure the shapes are right
    expected_shape = (update_size * batch_size,)
    if num_channels is not None:
        expected_shape = (num_channels,) + expected_shape
    assert stream.shape == (1,) + expected_shape
    assert new_snapshot.shape == snapshot.shape
    assert update_idx.item() == batch_size

    # now validate that the streamed value is correct
    start = size - update_size * (num_updates + batch_size - 1)
    start -= update_size * batch_size
    stop = start + update_size * batch_size
    expected = torch.arange(start, stop)
    if num_channels is None:
        assert (stream[0] == expected).all().item()
    else:
        for i in range(num_channels):
            assert (stream[0, i] == expected).all().item()

    # finally check that the snapshot values are as expected
    filled = (num_updates - 1) * update_size
    expected = torch.arange(stop, stop + filled)

    if num_channels is None:
        assert (new_snapshot[:filled] == expected).all().item()
        assert (new_snapshot[filled:] == 0).all().item()
    else:
        for i in range(num_channels):
            assert (new_snapshot[i, :filled] == expected).all().item()
            assert (new_snapshot[i, filled:] == 0).all().item()

    # now take the next step and confirm everything again
    stream, newer_snapshot, update_idx = averager(
        x[batch_size:], new_snapshot, update_idx
    )

    assert stream.shape == (1,) + expected_shape
    assert new_snapshot.shape == snapshot.shape
    assert update_idx.item() == 2 * batch_size

    start = size - update_size * (num_updates + batch_size - 1)
    stop = start + update_size * batch_size
    expected = torch.arange(start, stop)
    if num_channels is None:
        assert (stream[0] == expected).all().item()
    else:
        for i in range(num_channels):
            assert (stream[0, i] == expected).all().item()

    expected = torch.arange(stop, stop + filled)
    if num_channels is None:
        assert (newer_snapshot[:filled] == expected).all().item()
        assert (newer_snapshot[filled:] == 0).all().item()
    else:
        for i in range(num_channels):
            assert (newer_snapshot[i, :filled] == expected).all().item()
            assert (newer_snapshot[i, filled:] == 0).all().item()


@pytest.fixture(params=[None, 1, 2, 4])
def batch_size_with_None(request):
    return request.param


@pytest.mark.torch
def test_make_streaming_output_model(
    update_size, batch_size_with_None, num_updates, temp_local_repo
):
    from hermes.quiver.streaming.streaming_output import (
        make_streaming_output_model,
    )

    batch_size = batch_size_with_None
    input = MagicMock()
    input.shape = (batch_size or 8, 128)

    model = make_streaming_output_model(
        temp_local_repo, input, update_size, num_updates, batch_size=batch_size
    )

    batch_size = batch_size or 8

    assert len(model.config.input) == 1
    assert model.config.input[0].name == "update"
    assert model.config.input[0].dims == [batch_size, 128]

    assert len(model.config.output) == 1
    assert model.config.output[0].name == "output_stream"
    assert model.config.output[0].dims == [1, update_size * batch_size]

    assert len(model.config.sequence_batching.state) == 2
    assert model.config.sequence_batching.state[0].input_name == (
        "input_online_average"
    )
    assert model.config.sequence_batching.state[0].output_name == (
        "output_online_average"
    )
    assert model.config.sequence_batching.state[0].dims == [
        update_size * (batch_size + num_updates - 1)
    ]

    assert model.config.sequence_batching.state[1].input_name == (
        "input_update_index"
    )
    assert model.config.sequence_batching.state[1].output_name == (
        "output_update_index"
    )
    assert model.config.sequence_batching.state[1].dims == [1]

    input.shape = (None, 128)
    with pytest.raises(ValueError):
        make_streaming_output_model(
            temp_local_repo, input, update_size, num_updates, batch_size=None
        )

    input.shape = (batch_size - 1, 128)
    with pytest.raises(ValueError):
        make_streaming_output_model(
            temp_local_repo,
            input,
            update_size,
            num_updates,
            batch_size=batch_size,
        )
