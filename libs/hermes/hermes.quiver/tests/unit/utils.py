import torch
from hermes.quiver.io import LocalFileSystem


class IdentityModel(torch.nn.Module):
    def __init__(self, size: int = 10):
        super().__init__()
        self.W = torch.eye(size)

    def forward(self, x):
        return torch.matmul(x, self.W)


class DummyRepo:
    def __init__(self, fs_type=LocalFileSystem):
        self.fs_type = fs_type
        self.fs = None

    def __enter__(self):
        self.fs = self.fs_type("hermes-quiver-test")
        return self

    def __exit__(self, *exc_args):
        self.fs.delete()
