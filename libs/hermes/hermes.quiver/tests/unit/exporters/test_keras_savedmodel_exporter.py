import pytest

from hermes.quiver import Model, Platform
from hermes.quiver.exporters import KerasSavedModel


@pytest.mark.tensorflow
def test_keras_savedmodel_exporter(temp_local_repo, keras_model):
    scope = keras_model.name.split("_")[0]

    input_name = f"{scope}_dense_input"
    output_name = f"{scope}_dense/MatMul"
    assert keras_model.inputs[0].name.split(":")[0] == input_name
    assert keras_model.outputs[0].name.split(":")[0] == output_name

    model = Model("identity", temp_local_repo, Platform.ONNX)
    exporter = KerasSavedModel(model.config, model.fs)

    input_shapes = {input_name: (None, 10)}
    exporter._check_exposed_tensors("input", input_shapes)
    assert len(model.config.input) == 1
    assert model.config.input[0].name == input_name
    assert model.config.input[0].dims[0] == -1

    bad_input_shapes = {input_name: (None, 12)}
    with pytest.raises(ValueError):
        exporter._check_exposed_tensors("input", bad_input_shapes)

    output_shapes = exporter._get_output_shapes(keras_model, output_name)
    assert tuple(output_shapes[keras_model.layers[-1].name]) == (None, 10)

    exporter._check_exposed_tensors("output", output_shapes)
    assert len(model.config.output) == 1
    assert model.config.output[0].name == keras_model.layers[-1].name
    assert model.config.output[0].dims[0] == -1

    version_path = temp_local_repo.fs.join("identity", "1")
    output_path = temp_local_repo.fs.join(version_path, "model.savedmodel")
    temp_local_repo.fs.soft_makedirs(output_path)

    exporter.export(keras_model, output_path)

    # now test using full call
    exporter(keras_model, 2)
    with pytest.raises(ValueError):
        exporter(keras_model, 3, input_shapes)
    with pytest.raises(ValueError):
        exporter(keras_model, 3, None, ["y"])
