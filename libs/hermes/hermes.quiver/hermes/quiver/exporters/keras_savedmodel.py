import abc
import os
import tempfile

try:
    import tensorflow as tf

    _has_tf = True
except ImportError:
    _has_tf = False

from hermes.quiver import Platform
from hermes.quiver.exporters import Exporter


class KerasSavedModelMeta(abc.ABCMeta):
    @property
    def handles(self):
        if not _has_tf:
            raise ImportError(
                "Must have tensorflow installed to use "
                "KerasSavedModel exporter"
            )
        return tf.keras.Model

    @property
    def platform(self):
        return Platform.SAVEDMODEL


class KerasSavedModel(Exporter, metaclass=KerasSavedModelMeta):
    def __call__(
        self, model_fn, version, input_shapes=None, output_names=None
    ):
        if input_shapes is not None:
            raise ValueError(
                "Cannot specify input_shapes for KerasSavedModel exporter"
            )
        if output_names is not None:
            raise ValueError(
                "Cannot specify output_shapes for KerasSavedModel exporter"
            )

        if model_fn.inputs is None:
            # TODO: should we allow the specification of input
            # shapes if this is None and then create tensors to
            # map to output the way we do it in torch onnx? Presumably
            # you'll never want to be exporting a model that hasn't
            # been trained, so maybe this is overkill, but worth
            # thinking about
            raise ValueError(
                "No input shapes found, model hasn't been initialized"
            )

        # do this instead of grabbing names directly to
        # be robust to the case that a type_spec was specified
        input_shapes = {
            x._keras_history.layer.name: tuple(x.shape)
            for x in model_fn.inputs
        }

        super().__call__(model_fn, version, input_shapes, None)

    def _get_output_shapes(self, model_fn, output_names=None):
        output_shapes = {}
        layer_idx = {}
        for output in model_fn.outputs:
            name = output._keras_history.layer.name
            try:
                layer_idx[name] += 1
                name += "_" + str(layer_idx[name])
            except KeyError:
                layer_idx[name] = 0

            output_shapes[name] = output.shape

        return output_shapes

    def export(self, model_fn, export_path, verbose=0):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_fn.save(tmpdir)

            # write all the objects in manually
            # TODO: implement a `copy` functionality
            for root, ds, fs in os.walk(tmpdir):
                # create any directories we might need
                for d in ds:
                    path = os.path.join(root, d)
                    write_dir = path.replace(tmpdir, "").split(os.path.sep)
                    write_dir = self.fs.join(
                        *([export_path] + [i for i in write_dir if i])
                    )
                    self.fs.soft_makedirs(write_dir)

                # read them into memory as bytes
                for f in fs:
                    path = os.path.join(root, f)
                    with open(path, "rb") as f:
                        stuff = f.read()

                    # format the write path
                    write_path = path.replace(tmpdir, "").split(os.path.sep)
                    write_path = self.fs.join(
                        *([export_path] + [i for i in write_path if i])
                    )

                    # write the bytes to the filesystem
                    self.fs.write(stuff, write_path)
