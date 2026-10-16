from unittest.mock import patch

import pytest

from hermes.quiver import Model, Platform
from hermes.quiver.exporters import TorchTensorRT


@pytest.mark.torch
def test_torch_tensorrt_exporter_with_dynamo(temp_local_repo, torch_model):
    """A model from torch.export can't be exported with the TorchScript
    exporter, so building its intermediate ONNX model needs dynamo."""
    import torch

    program = torch.export.export(
        torch_model,
        (torch.randn(2, 10),),
        dynamic_shapes=({0: torch.export.Dim.AUTO},),
    )
    model = Model("identity", temp_local_repo, Platform.TENSORRT)
    exporter = TorchTensorRT(model.config, model.fs)
    version_path = temp_local_repo.fs.join("identity", "1")
    temp_local_repo.fs.soft_makedirs(version_path)

    # stand in for the TensorRT conversion, which needs a GPU
    with patch.object(TorchTensorRT, "convert_remote", return_value=b"trt"):
        exporter(
            program.module(),
            1,
            input_shapes={"x": (None, 10)},
            output_names=["y"],
            endpoint="unused",
            dynamo=True,
        )
