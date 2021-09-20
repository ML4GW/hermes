import os
import sys

import pytest

from hermes.quiver import Model, Platform
from hermes.quiver.exporters import TorchOnnx

sys.path.insert(0, os.path.dirname(__file__))
from utils import DummyRepo, IdentityModel  # noqa


def test_model():
    with DummyRepo() as repo:
        model = Model("test", repo, platform=Platform.ONNX)

        assert os.path.exists(os.path.join("hermes-quiver-test", model.name))
        assert len(model.versions) == 0
        assert len(model.inputs) == 0
        assert len(model.outputs) == 0

        model_fn = IdentityModel()
        assert isinstance(model._find_exporter(model_fn), TorchOnnx)

        export_path = model.export_version(
            model_fn, input_shapes={"x": [None, 10]}, output_names=["y"]
        )
        assert export_path == repo.fs.join(model.name, "1", "model.onnx")
        assert len(model.versions) == 1
        assert model.config.input[0].name == "x"
        assert list(model.config.input[0].dims) == [-1, 10]
        assert model.inputs["x"].shape == (None, 10)

        with pytest.raises(ValueError):
            export_path = model.export_version(
                model_fn, input_shapes={"x": [None, 12]}
            )
        assert len(model.versions) == 1

        export_path = model.export_version(model_fn)
        assert export_path == repo.fs.join(model.name, "2", "model.onnx")

        repo.fs.remove(repo.fs.join(model.name, "1"))
        export_path = model.export_version(model_fn)
        assert export_path == repo.fs.join(model.name, "3", "model.onnx")
        assert len(model.versions) == 2
