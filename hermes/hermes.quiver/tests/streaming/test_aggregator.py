import numpy as np
import pytest


@pytest.mark.tensorflow
@pytest.fixture(params=[(1, 1), (1, 2), (2, 4), (10, 8)])
def aggregator(request):
    from hermes.quiver.streaming.streaming_output import Aggregator

    num_updates, update_size = request.param
    return Aggregator(num_updates=num_updates, update_size=update_size)


@pytest.mark.tensorflow
def test_aggregator(aggregator):
    num_updates = aggregator.num_updates
    update_size = aggregator.update_size

    new_seq = np.array([1.0])
    old_seq = np.array([0.0])

    # make sure that our shape checks raise the appropriate errors
    with pytest.raises(ValueError):
        # must have batch size of 1
        x = np.ones((2, num_updates * update_size))
        aggregator(x, new_seq)

    with pytest.raises(ValueError):
        # must have the at least appropriate update size
        x = np.ones((1, update_size * (num_updates - 1)))
        aggregator(x, new_seq)

    # run an input through the layer so that
    # it gets built, indicating that this is
    # the start of a new sequence
    x = np.arange(num_updates * update_size)[None].astype("float32")

    # make sure that the update index is set to 1
    # and that the output matches the first step of the input
    y = aggregator(x, new_seq)
    assert aggregator.update_idx.numpy() == 1
    assert y.shape == (1, update_size)
    assert (y.numpy() == np.arange(update_size)).all()

    # now run another input through on this sequence,
    # with the value incremented by one,
    # and make sure that the update index increments
    # and that the output is the average of the
    # first and second inputs
    y = aggregator(x, old_seq)
    if num_updates > 1:
        expected = np.arange(update_size / 2, update_size / 2 + update_size)
    else:
        expected = np.arange(update_size)
    assert aggregator.update_idx.numpy()
    assert (y.numpy() == expected).all()

    # verify once more that we have the average of all 3
    y = aggregator(x, old_seq)
    if num_updates > 2:
        expected = np.arange(update_size, 2 * update_size)
    assert (y.numpy() == expected).all()

    # now restart the sequence and make sure that
    # everything resets properly
    y = aggregator(x + 1, new_seq)
    assert aggregator.update_idx.numpy() == 1
    assert (y.numpy() == np.arange(1, update_size + 1)).all()
