import pytest


@pytest.mark.torch
def test_utils(temp_local_repo):
    import torch

    from hermes.quiver.streaming import utils

    class DummyLayer(torch.nn.Module):
        def forward(self, x, state):
            return x + state, state + 1

    model = utils.add_streaming_model(
        temp_local_repo,
        streaming_layer=DummyLayer(),
        name="dummy",
        input_name="dummy_input",
        input_shape=(4, 10),
        state_names=["dummy_state"],
        state_shapes=[(4, 10)],
        output_names=["dummy_output"],
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
    assert config.input[0].dims == [4, 10]

    assert len(config.sequence_batching.state) == 1
    assert config.sequence_batching.state[0].input_name == "input_dummy_state"
    assert config.sequence_batching.state[0].output_name == (
        "output_dummy_state"
    )
    assert config.sequence_batching.state[0].dims == [4, 10]

    assert len(config.output) == 1
    assert config.output[0].name == "dummy_output"
    assert config.output[0].dims == [4, 10]
