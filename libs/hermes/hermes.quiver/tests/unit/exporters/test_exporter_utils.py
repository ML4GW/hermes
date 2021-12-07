from unittest.mock import Mock

import pytest

from hermes.quiver.exporters import KerasSavedModel, TorchOnnx, utils
from hermes.quiver.platform import Platform


def make_model(platform):
    model = Mock()
    model.platform = platform
    model.fs = Mock()
    model.config = Mock()
    return model


def _test_find_exporter(model_fn, good_model, bad_model, expected):
    exporter = utils.find_exporter(model_fn, good_model)
    assert isinstance(exporter, expected)
    with pytest.raises(TypeError):
        utils.find_exporter(model_fn, bad_model)


@pytest.mark.tensorflow
def test_find_tensorflow_exporter(keras_model):
    good_model = make_model(Platform.SAVEDMODEL)
    bad_model = make_model(Platform.ONNX)
    _test_find_exporter(keras_model, good_model, bad_model, KerasSavedModel)


@pytest.mark.torch
def test_find_torch_exporter(torch_model):
    good_model = make_model(Platform.ONNX)
    bad_model = make_model(Platform.SAVEDMODEL)
    _test_find_exporter(torch_model, good_model, bad_model, TorchOnnx)
