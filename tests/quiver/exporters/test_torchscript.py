import pytest

from hermes.quiver import Model, Platform
from hermes.quiver.exporters import TorchScript


@pytest.mark.torch
def test_torchscript_exporter(temp_local_repo, torch_model):
    model_fn = torch_model

    model = Model("identity", temp_local_repo, Platform.TORCHSCRIPT)
    exporter = TorchScript(model.config, model.fs)

    input_shapes = {"x": (None, 10)}
    exporter._check_exposed_tensors("input", input_shapes)
    assert len(model.config.input) == 1
    assert model.config.input[0].name == "x"
    assert model.config.input[0].dims[0] == -1

    bad_input_shapes = {"x": (None, 12)}
    with pytest.raises(ValueError):
        exporter._check_exposed_tensors("input", bad_input_shapes)

    output_shapes = exporter._get_output_shapes(model_fn, "y")
    assert output_shapes["OUTPUT__0"] == (None, 10)

    exporter._check_exposed_tensors("output", output_shapes)
    assert len(model.config.output) == 1
    assert model.config.output[0].name == "OUTPUT__0"
    assert model.config.output[0].dims[0] == -1

    version_path = temp_local_repo.fs.join("identity", "1")
    temp_local_repo.fs.soft_makedirs(version_path)
    output_path = temp_local_repo.fs.join(version_path, "model.torch")
    exporter.export(model_fn, output_path)

    # now test full __call__ method
    model2 = Model("identity2", temp_local_repo, Platform.TORCHSCRIPT)
    exporter = TorchScript(model2.config, model2.fs)

    model_path = temp_local_repo.fs.root
    with pytest.raises(ValueError):
        exporter(model_fn, model_path, [(None, 10)], ["BAD_NAME"])
    exporter(model_fn, model_path, [(None, 10)])
    assert "INPUT__0" in model2.inputs
    assert "OUTPUT__0" in model2.outputs
