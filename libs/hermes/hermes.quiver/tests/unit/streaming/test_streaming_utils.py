import pytest


@pytest.mark.tensorflow
def test_utils(temp_local_repo, tf):
    from hermes.quiver.streaming import utils

    class DummyLayer(tf.keras.layers.Layer):
        def call(self, x, sequence_start):
            return x + (1 * sequence_start)

    model = utils.add_streaming_model(
        temp_local_repo,
        streaming_layer=DummyLayer(),
        name="dummy",
        input_name="dummy_input",
        input_shape=(4, 10),
        streams_per_gpu=2,
    )

    assert model.name == "dummy"
    assert "dummy" in temp_local_repo.fs.list()

    config = temp_local_repo.fs.read_config(
        temp_local_repo.fs.join("dummy", "config.pbtxt")
    )
    assert config.name == "dummy"

    assert len(config.input) == 1
    assert config.input[0].name == "dummy_input"
    assert config.input[0].dims == [1, 4, 10]

    assert config.sequence_batching.direct is not None
    assert len(config.sequence_batching.control_input) == 1
    assert config.sequence_batching.control_input[0].name == "sequence_start"

    assert config.instance_group[0].count == 2
