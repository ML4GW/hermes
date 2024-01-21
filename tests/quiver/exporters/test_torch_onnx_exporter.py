from unittest.mock import ANY, patch

import pytest

from hermes.quiver import Model, Platform
from hermes.quiver.exporters import TorchOnnx


@pytest.mark.torch
def test_torch_onnx_exporter(temp_local_repo, torch_model):
    model_fn = torch_model

    model = Model("identity", temp_local_repo, Platform.ONNX)
    exporter = TorchOnnx(model.config, model.fs)

    input_shapes = {"x": (None, 10)}
    exporter._check_exposed_tensors("input", input_shapes)
    assert len(model.config.input) == 1
    assert model.config.input[0].name == "x"
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
    output_path = temp_local_repo.fs.join(version_path, "model.onnx")
    exporter.export(model_fn, output_path)
    # TODO: include onnx as dev dependency for checking

    # now test using list-style input passing
    model2 = Model("identity2", temp_local_repo, Platform.ONNX)
    exporter = TorchOnnx(model2.config, model2.fs)

    input_shapes = [(None, 10)]
    exporter._check_exposed_tensors("input", input_shapes)
    assert len(model2.config.input) == 1
    assert model2.config.input[0].name == "input_0"


@pytest.mark.torch
def test_torch_onnx_exporter_with_kwargs(temp_local_repo, torch_model):
    model_fn = torch_model

    model = Model("test", temp_local_repo, Platform.ONNX)
    exporter = TorchOnnx(model.config, model.fs)

    version_path = temp_local_repo.fs.join("test", "1")
    temp_local_repo.fs.soft_makedirs(version_path)
    output_path = temp_local_repo.fs.join(version_path, "model.onnx")

    with patch("hermes.quiver.exporters.torch_onnx.torch.onnx.export") as mock:
        exporter.export(model_fn, output_path, opset_version=11)
        mock.assert_called_with(
            model_fn,
            ANY,
            ANY,
            input_names=ANY,
            output_names=ANY,
            dynamic_axes=ANY,
            opset_version=11,
        )
