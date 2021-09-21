import os
import random
import sys

import pytest

from hermes.quiver import Model, Platform
from hermes.quiver.exporters import KerasSavedModel
from hermes.quiver.io import GCSFileSystem, LocalFileSystem

sys.path.insert(0, os.path.dirname(__file__))
from utils import DummyRepo, IdentityKerasModel  # noqa


@pytest.mark.parametrize("fs_type", [LocalFileSystem, GCSFileSystem])
def test_torch_onnx_exporter(fs_type):
    scope = "".join(random.choices("abcdefghijk", k=10))
    model_fn = IdentityKerasModel(size=10, scope=scope)

    input_name = f"{scope}_dense_input"
    output_name = f"{scope}_dense/MatMul"
    assert model_fn.inputs[0].name.split(":")[0] == input_name
    assert model_fn.outputs[0].name.split(":")[0] == output_name

    with DummyRepo(fs_type) as repo:
        model = Model("identity", repo, Platform.ONNX)
        exporter = KerasSavedModel(model.config, model.fs)

        input_shapes = {input_name: (None, 10)}
        exporter._check_exposed_tensors("input", input_shapes)
        assert len(model.config.input) == 1
        assert model.config.input[0].name == input_name
        assert model.config.input[0].dims[0] == -1

        bad_input_shapes = {input_name: (None, 12)}
        with pytest.raises(ValueError):
            exporter._check_exposed_tensors("input", bad_input_shapes)

        output_shapes = exporter._get_output_shapes(model_fn, output_name)
        assert output_shapes[output_name] == (None, 10)

        exporter._check_exposed_tensors("output", output_shapes)
        assert len(model.config.output) == 1
        assert model.config.output[0].name == output_name
        assert model.config.output[0].dims[0] == -1

        version_path = repo.fs.join("identity", "1")
        output_path = repo.fs.join(version_path, "model.savedmodel")
        repo.fs.soft_makedirs(output_path)

        exporter.export(model_fn, output_path)

        # now test using full call
        exporter(model_fn, 2)
        with pytest.raises(ValueError):
            exporter(model_fn, 3, input_shapes)
        with pytest.raises(ValueError):
            exporter(model_fn, 3, None, ["y"])
