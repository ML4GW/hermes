import os
import sys

import pytest

from hermes.quiver import Model, Platform
from hermes.quiver.exporters import TorchOnnx
from hermes.quiver.io import GCSFileSystem, LocalFileSystem

sys.path.insert(0, os.path.dirname(__file__))
from utils import DummyRepo, IdentityTorchModel  # noqa


@pytest.mark.parametrize("fs_type", [LocalFileSystem, GCSFileSystem])
def test_torch_onnx_exporter(fs_type):
    model_fn = IdentityTorchModel()

    with DummyRepo(fs_type) as repo:
        model = Model("identity", repo, Platform.ONNX)
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

        version_path = repo.fs.join("hermes-quiver-test", "identity", "1")
        repo.fs.soft_makedirs(version_path)

        output_path = repo.fs.join(version_path, "model.onnx")

        exporter.export(model_fn, output_path)
        # TODO: include onnx as dev dependency for checking

        # now test using list-style input passing
        model2 = Model("identity2", repo, Platform.ONNX)
        exporter = TorchOnnx(model2.config, model2.fs)

        input_shapes = [(None, 10)]
        exporter._check_exposed_tensors("input", input_shapes)
        assert len(model2.config.input) == 1
        assert model2.config.input[0].name == "input_0"
