from unittest.mock import Mock

import pytest
from tritonclient.grpc.model_config_pb2 import ModelInstanceGroup

from hermes.quiver import ModelConfig, Platform


def test_model_config(temp_fs):
    model = Mock()
    model.fs = temp_fs
    model.name = "test-model"
    model.platform = Platform.ONNX
    temp_fs.soft_makedirs(model.name)

    # make sure we can't pass name or
    # platform
    for bad_kwarg in ["name", "platform"]:
        with pytest.raises(ValueError):
            kwargs = {bad_kwarg: "bad"}
            config = ModelConfig(model, **kwargs)

    # build a config and make sure it
    # has all the appropriate attributes
    config = ModelConfig(model, max_batch_size=8)
    assert config.name == "test-model"
    assert config.platform == "onnxruntime_onnx"
    assert config.max_batch_size == 8

    # add an input and make sure
    # that it exists on the config
    # and has the appropriate properties
    input = config.add_input(
        name="test_input", shape=(None, 8), dtype="float32"
    )
    assert len(config.input) == 1
    assert config.input[0] == input
    assert config.input[0].name == "test_input"

    # add an instance group on to the
    # config and make sure it exists
    # and has all the appropriate properties
    instance_group = config.add_instance_group(kind="gpu", gpus=2, count=4)
    assert len(config.instance_group) == 1
    assert instance_group == config.instance_group[0]
    assert config.instance_group[0].kind == ModelInstanceGroup.KIND_GPU
    assert config.instance_group[0].gpus == [0, 1]
    assert config.instance_group[0].count == 4

    # change a value on the instance group
    # and ensure it's relfected on the config
    config.instance_group[0].count = 6
    assert config.instance_group[0].count == 6

    # write the config to the repo
    config.write()

    # first make sure that the the config
    # gotten written properly to the right
    # place
    config_path = model.fs.join(config.name, "config.pbtxt")
    new_config = model.fs.read_config(config_path)
    assert new_config.SerializeToString() == config.SerializeToString()

    # next initialize a new config using
    # the existing model to see if it loads
    # things in properly
    new_config = ModelConfig(model)
    assert new_config.SerializeToString() == config.SerializeToString()

    # finally check to make sure that kwargs
    # provided with initialization override
    # those in the existing config
    new_config = ModelConfig(model, max_batch_size=4)
    assert new_config.max_batch_size == 4
    assert new_config.input[0].name == "test_input"
