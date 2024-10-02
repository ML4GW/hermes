import pytest

from hermes.quiver import Model, Platform
from hermes.quiver.exporters import TorchScript


@pytest.mark.torch
def test_torchscript_exporter(temp_local_repo, torch_model, torch_model_2):
    model_fn = torch_model

    model = Model("identity", temp_local_repo, Platform.TORCHSCRIPT)
    exporter = TorchScript(model.config, model.fs)

    input_shapes = {"input0": (None, 10)}
    exporter._check_exposed_tensors("input", input_shapes)
    assert len(model.config.input) == 1
    assert model.config.input[0].name == "input0"
    assert model.config.input[0].dims[0] == -1

    bad_input_shapes = {"x": (None, 12)}
    with pytest.raises(ValueError):
        exporter._check_exposed_tensors("input", bad_input_shapes)

    output_shapes = exporter._get_output_shapes(model_fn, "y")
    assert output_shapes["y"] == (None, 10)

    exporter._check_exposed_tensors("output", output_shapes)
    assert len(model.config.output) == 1
    assert model.config.output[0].name == "y"
    assert model.config.output[0].dims[0] == -1

    version_path = temp_local_repo.fs.join("identity", "1")
    temp_local_repo.fs.soft_makedirs(version_path)
    output_path = temp_local_repo.fs.join(version_path, "model.torch")
    exporter.export(model_fn, output_path)

    # now test full __call__ method
    model2 = Model("identity2", temp_local_repo, Platform.TORCHSCRIPT)
    exporter = TorchScript(model2.config, model2.fs)

    model_path = temp_local_repo.fs.root

    # if a dictionary if input_shapes is not passed
    # (i.e. no user specified name) it should default
    # to the name of the input tensor in the forward call of the model
    # in this case, "x"
    exporter(model_fn, model_path, [(None, 10)])
    assert "x" in model2.inputs
    assert "OUTPUT__0" in model2.outputs

    # if a dictionary of input_shapes is passed, it should
    # use the user specified names
    model3 = Model("identity3", temp_local_repo, Platform.TORCHSCRIPT)
    exporter = TorchScript(model3.config, model3.fs)
    model_path = temp_local_repo.fs.root

    exporter(model_fn, model_path, input_shapes={"my_name": (None, 10)})
    assert "my_name" in model3.inputs
    assert "OUTPUT__0" in model3.outputs

    # now check using non-default output names
    model4 = Model("identity4", temp_local_repo, Platform.TORCHSCRIPT)
    exporter = TorchScript(model4.config, model4.fs)
    model_path = temp_local_repo.fs.root

    exporter(
        model_fn,
        model_path,
        input_shapes={"my_name": (None, 10)},
        output_names=["my_output"],
    )
    assert "my_name" in model4.inputs
    assert "my_output" in model4.outputs

    # test a model with multiple inputs and outputs
    model_fn = torch_model_2

    model = Model("identity", temp_local_repo, Platform.TORCHSCRIPT)
    exporter = TorchScript(model.config, model.fs)

    input_shapes = {"input0": (None, 10), "input1": (None, 10)}
    exporter._check_exposed_tensors("input", input_shapes)
    assert len(model.config.input) == 2
    assert model.config.input[0].name == "input0"
    assert model.config.input[1].name == "input1"
    assert model.config.input[0].dims[0] == -1
    assert model.config.input[1].dims[0] == -1

    bad_input_shapes = {"x": (None, 12)}
    with pytest.raises(ValueError):
        exporter._check_exposed_tensors("input", bad_input_shapes)

    output_shapes = exporter._get_output_shapes(
        model_fn, ["output1", "output2"]
    )
    assert output_shapes["output1"] == (None, 10)
    assert output_shapes["output2"] == (None, 10)

    model4 = Model("identity4", temp_local_repo, Platform.TORCHSCRIPT)
    exporter = TorchScript(model4.config, model4.fs)
    model_path = temp_local_repo.fs.root

    exporter(model_fn, model_path, input_shapes=[(None, 10), (None, 10)])
    assert "x" in model4.inputs and "y" in model4.inputs
    assert "OUTPUT__0" in model4.outputs and "OUTPUT__1" in model4.outputs
