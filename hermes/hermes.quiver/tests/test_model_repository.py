import pytest

from hermes.quiver import ModelRepository, Platform
from hermes.quiver.io import LocalFileSystem


def test_model_repository(temp_fs):
    # make sure clean functionality works with no models
    repo = ModelRepository(str(temp_fs), clean=True)
    assert len(repo.models) == 0

    repo = ModelRepository(str(temp_fs), clean=False)
    assert len(repo.models) == 0

    model = repo.add("test-model", platform=Platform.ONNX)
    assert model.name == "test-model"
    assert model.platform == Platform.ONNX
    assert len(model.config.input) == 0
    assert len(model.config.output) == 0
    assert len(model.versions) == 0

    # creating the same model without force raises an error
    with pytest.raises(ValueError):
        model = repo.add("test-model", platform=Platform.ONNX)

    # make sure index gets appended
    model = repo.add("test-model", platform=Platform.ONNX, force=True)
    assert model.name == "test-model_0"

    # remove then re-add the model to make sure it gets
    # the appropriate name
    repo.remove("test-model")
    assert len(repo.models) == 1
    model = repo.add("test-model", platform=Platform.ONNX)
    assert model.name == "test-model"
    assert len(repo.models) == 2

    # now delete a model externally and make sure
    # that the refresh recreates the correct state
    temp_fs.remove("test-model_0")
    assert len(repo.models) == 2

    with pytest.raises(ValueError):
        # this will fail since test-model hasn't
        # written a config yet
        repo.refresh()

    model.config.write()
    repo.refresh()
    assert len(repo.models) == 1

    # make sure that recreating the repo initializes
    # the models correctly
    repo = ModelRepository(str(temp_fs))
    assert len(repo.models) == 1

    # now make sure the clean functionality works with a model in the repo
    repo = ModelRepository(str(temp_fs), clean=True)
    assert len(repo.models) == len(temp_fs.list("")) == 0

    # finally make sure the repo deletion works
    model = repo.add("test-model", platform=Platform.ONNX)
    assert len(repo.models) == len(temp_fs.list("")) == 1

    repo.delete()
    assert len(repo.models) == 0

    # TODO: what's the check for GCSFileSystem
    if isinstance(temp_fs, LocalFileSystem):
        with pytest.raises(FileNotFoundError):
            temp_fs.list("")
