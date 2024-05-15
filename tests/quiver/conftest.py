import random
from unittest.mock import Mock

import pytest

from hermes.quiver import ModelRepository
from hermes.quiver.io import GCSFileSystem, LocalFileSystem
from hermes.quiver.io.exceptions import NoFilesFoundError


@pytest.fixture(
    scope="module",
    params=[
        LocalFileSystem,
        pytest.param(GCSFileSystem, marks=pytest.mark.gcs),
    ],
)
def fs(request):
    return request.param


@pytest.fixture(scope="module")
def temp_fs(fs):
    filesystem = fs("hermes-quiver-test")
    yield filesystem

    try:
        filesystem.delete()
    except NoFilesFoundError:
        pass


@pytest.fixture(scope="module")
def temp_dummy_repo(temp_fs):
    repo = Mock()
    repo.fs = temp_fs
    return repo


@pytest.fixture(scope="module")
def temp_local_repo():
    repo = ModelRepository("hermes-quiver-test-local")
    yield repo
    repo.delete()


@pytest.fixture(scope="module")
def tf():
    import tensorflow as tf

    return tf


@pytest.fixture(scope="module", params=[10])
def dim(request):
    return request.param


@pytest.fixture
def keras_model(dim, tf):
    import tensorflow as tf

    scope = "".join(random.choices("abcdefghijk", k=10))
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                dim,
                use_bias=False,
                kernel_initializer="identity",
                name=f"{scope}_dense",
            )
        ],
        name=scope,
    )

    # do a couple batch sizes to get variable size
    for batch_size in range(1, 3):
        y = model(tf.ones((batch_size, dim)))
        assert (y.numpy() == 1.0).all()
    return model


@pytest.fixture
def torch_model(dim):
    import torch

    class Model(torch.nn.Module):
        def __init__(self, size: int = 10):
            super().__init__()
            self.size = size
            self.W = torch.eye(size)

        def forward(self, x):
            return torch.matmul(x, self.W)

    return Model(dim)
