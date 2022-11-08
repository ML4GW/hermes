import pickle
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

import requests

try:
    from hermes.quiver.exporters.tensorrt.onnx import convert_network

    _has_trt = True
except ImportError as e:
    _cuda_error = "cannot open shared object file"
    if "tensorrt" not in str(e) and _cuda_error not in str(e):
        raise
    _has_trt = False
    _error_msg = str(e)

from hermes.quiver.exporters.torch_onnx import TorchOnnx, TorchOnnxMeta
from hermes.quiver.platform import Platform, conventions
from hermes.quiver.types import SHAPE_TYPE

if TYPE_CHECKING:
    from hermes.quiver.model import Model


class TorchTensorRTMeta(TorchOnnxMeta):
    @property
    def handles(self):
        # have to import Model here since model depends
        # on exporter utils, which imports this
        from hermes.quiver.model import Model

        handle = TorchOnnxMeta.handles.fget(self)
        return (handle, Model)

    @property
    def platform(self):
        return Platform.TENSORRT


class TorchTensorRT(TorchOnnx, metaclass=TorchTensorRTMeta):
    def convert_local(self, model_binary: bytes, use_fp16: bool) -> bytes:
        # do the conversion locally
        trt_binary = convert_network(
            model_binary, self.config._config, use_fp16
        )

        # CUDA engine build won't raise an error on failure
        # but will return None instead, so raise error here
        if trt_binary is None:
            raise RuntimeError("Model conversion failed, consult TRT logs")
        return bytes(trt_binary)

    def convert_remote(
        self, model_binary: bytes, use_fp16: bool, endpoint: str
    ) -> bytes:
        # use a remote conversion service to convert
        # the onnx binary to a tensorrt one
        data = {
            "config": self.config._config.SerializeToString(),
            "network": model_binary,
            "use_fp16": use_fp16,
        }
        response = requests.post(
            url=endpoint,
            data=pickle.dumps(data),
            headers={"Content-Type": "application/octet-stream"},
        )

        response.raise_for_status()
        return response.content

    def __call__(
        self,
        model_fn: Union[Callable, "Model"],
        version: int,
        input_shapes: Optional[SHAPE_TYPE] = None,
        output_names: Optional[Sequence[str]] = None,
        use_fp16: bool = False,
        endpoint: Optional[str] = None,
    ):
        if endpoint is None and not _has_trt:
            raise ImportError(
                "Must have  tensorrt installed to use TorchTensorRT "
                "export platform if no conversion endpoint is specified. "
                "Encountered import error: '{}'".format(_error_msg)
            )

        def do_conversion(model_binary, config):
            # merge info about inputs and outputs from
            # the existing config into our config
            inputs = {i.name: tuple(i.dims) for i in config.input}
            outputs = {i.name: tuple(i.dims) for i in config.output}
            self._check_exposed_tensors("input", inputs)
            self._check_exposed_tensors("output", outputs)

            if endpoint is None:
                trt_binary = self.convert_local(model_binary, use_fp16)
            else:
                trt_binary = self.convert_remote(
                    model_binary, use_fp16, endpoint
                )

            # write the binary file to the appropriate location
            export_path = self.fs.join(
                self.config.name, str(version), conventions[self.platform]
            )
            self.fs.write(trt_binary, export_path)
            return export_path

        if isinstance(model_fn, self.handles[0]):
            # if we didn't pass a model, then create a dummy
            # temporary model repository to build the ONNX
            # binary that we'll convert and do export with that

            # infer shapes and names if we didn't pass them explicitly
            if input_shapes is None and len(self.config.input) == 0:
                raise ValueError(
                    "Must specify input shapes if passing "
                    "a torch model for export directly"
                )
            elif input_shapes is None:
                input_shapes = {
                    i.name: list(i.dims) for i in self.config.input
                }

            if output_names is None and len(self.config.output) == 0:
                raise ValueError(
                    "Must specify output names if passing "
                    "a torch model for export directly"
                )
            elif output_names is None:
                output_names = [i.name for i in self.config.output]

            # use temporary directory as context so
            # that it will delete no matter what happens
            with TemporaryDirectory() as d:
                # do import here since model repo depends on
                # model which depends on exporter
                # TODO: what's a better way to handle this?
                from hermes.quiver.model_repository import ModelRepository

                repo = ModelRepository(d)
                model = repo.add("tmp", platform=Platform.ONNX)
                onnx_path = model.export_version(
                    model_fn,
                    version=version,
                    input_shapes=input_shapes,
                    output_names=output_names,
                )
                onnx_path = repo.fs.join(d, onnx_path)
                with open(onnx_path, "rb") as f:
                    model_binary = f.read()

                return do_conversion(model_binary, model.config._config)
        else:
            onnx_path = model_fn.repository.fs.join(
                model_fn.name, str(version), conventions[self.platform]
            )
            with open(onnx_path, "rb") as f:
                model_binary = f.read()

            return do_conversion(model_binary, model_fn.config._config)
