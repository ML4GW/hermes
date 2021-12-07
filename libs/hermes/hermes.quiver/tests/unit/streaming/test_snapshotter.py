from collections import OrderedDict

import numpy as np
import pytest


@pytest.fixture(params=[1, 4, 100])
def snapshot_size(request):
    return request.param


@pytest.fixture
def channels():
    return OrderedDict([("a", 6), ("b", 4), ("c", 8)])


@pytest.fixture
def snapshotter(snapshot_size, channels):
    from hermes.quiver.streaming.streaming_input import Snapshotter

    return Snapshotter(snapshot_size, channels)


@pytest.mark.tensorflow
@pytest.mark.parametrize("update_size", [1, 10])
def test_snapshotter(snapshotter, update_size):
    channel_vals = snapshotter.channels.values()
    num_channels = sum(channel_vals)

    # make sure our shape checks catch any inconsistencies
    with pytest.raises(ValueError):
        # channel dimension must equal the total number of input channels
        x = np.ones((1, num_channels - 1, update_size))
        snapshotter(x, 1)

    with pytest.raises(ValueError):
        # can't support batching
        x = np.ones((2, num_channels, update_size))
        snapshotter(x, 1)

    # now run an input through as a new sequence and
    # make sure we get the appropriate number of outputs
    # if our update size is too large, this should raise
    # an error then we're done testing this combination
    x = np.ones((1, num_channels, update_size))
    if update_size > snapshotter.snapshot_size:
        with pytest.raises(ValueError):
            y = snapshotter(x, 1)
        return

    y = snapshotter(x, 1)
    assert len(y) == len(channel_vals)

    # now make sure that each snapshot has the appropriate shape
    # and has 0s everywhere except for the most recent update
    for y_, channels in zip(y, channel_vals):
        y_ = y_.numpy()
        assert y_.shape == (1, channels, snapshotter.snapshot_size)
        assert (y_[:, :, :-update_size] == 0).all()
        assert (y_[:, :, -update_size:] == 1).all()

    # make another update and verify that the snapshots
    # all contain the expected update values
    y = snapshotter(x + 1, 0)
    for y_ in y:
        y_ = y_.numpy()
        assert (y_[:, :, : -update_size * 2] == 0).all()
        for i in range(2):
            start = -update_size * (2 - i)
            stop = (-update_size * (1 - i)) or None
            assert (y_[:, :, start:stop] == i + 1).all()

    # reset the sequence and make sure that the
    # snapshot resets and updates properly
    y = snapshotter(x + 2, 1)
    for y_ in y:
        y_ = y_.numpy()
        assert (y_[:, :, :-update_size] == 0).all()
        assert (y_[:, :, -update_size:] == 3).all()
