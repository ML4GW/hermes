import tensorflow as tf
import torch

from hermes.quiver.io import LocalFileSystem


class IdentityTorchModel(torch.nn.Module):
    def __init__(self, size: int = 10):
        super().__init__()
        self.W = torch.eye(size)

    def forward(self, x):
        return torch.matmul(x, self.W)


def IdentityKerasModel(size: int, scope: str):
    model_fn = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                size,
                use_bias=False,
                kernel_initializer="identity",
                name=f"{scope}_dense",
            )
        ]
    )

    # do a couple batch sizes to get variable size
    for batch_size in range(1, 3):
        y = model_fn(tf.ones((batch_size, size)))
        assert (y.numpy() == 1.0).all()
    return model_fn


class DummyRepo:
    def __init__(self, fs_type=LocalFileSystem):
        self.fs_type = fs_type
        self.fs = None

    def __enter__(self):
        self.fs = self.fs_type("hermes-quiver-test")
        return self

    def __exit__(self, *exc_args):
        self.fs.delete()
