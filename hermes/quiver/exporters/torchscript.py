import abc
from collections import OrderedDict
from io import BytesIO

try:
    import torch

    _has_torch = True
except ImportError:
    _has_torch = False

from hermes.quiver import Platform
from hermes.quiver.exporters import Exporter
from hermes.quiver.exporters.utils import get_input_names_from_torch_object


class TorchScriptMeta(abc.ABCMeta):
    @property
    def handles(self):
        if not _has_torch:
            raise ImportError(
                "Must have torch installed to use TorchScript export platform"
            )
        return torch.nn.Module

    @property
    def platform(self):
        return Platform.TORCHSCRIPT


class TorchScript(Exporter, metaclass=TorchScriptMeta):
    def __call__(
        self, model_fn, version, input_shapes, output_names=None
    ) -> None:
        if output_names is not None:
            raise ValueError(
                "Cannot specify output_names for TorchScript exporter"
            )

        input_shapes = {f"INPUT__{i}": j for i, j in enumerate(input_shapes)}
        super().__call__(model_fn, version, input_shapes, output_names)

    def _get_tensor(self, shape):
        tensor_shape = []
        for dim in shape:
            if dim == -1:
                dim = self.config.max_batch_size or 1
            tensor_shape.append(dim)

        # TODO: this will not always be safe, for
        # example if your inputs need to be in a
        # certain range or are e.g. categorical
        # Should we expose an optional `input_tensors`
        # argument? We could accept framework tensors
        # at the `input_shapes` arg of `Platform.export`
        # and pass them along if they were provided?
        return torch.randn(*tensor_shape)

    def _get_output_shapes(self, model_fn, _):
        # now that we know we have inputs added to our
        # model config, use that config to generate
        # framework tensors that we'll feed through
        # the network to inspect the output
        input_tensors = OrderedDict()
        for input in self.config.input:
            # generate an input array of random data
            input_tensors[input.name] = self._get_tensor(input.dims)

        # parse either a `ScriptModule` or `torch.nn.Module`
        # to figure out in which order to pass input tensors to the model_fn
        parameters = get_input_names_from_torch_object(model_fn)

        # make sure the number of inputs to
        # the model_fn matches the number of
        # specified input tensors
        if len(parameters) != len(input_tensors):
            raise ValueError(
                "Model function  expects {} inputs, but "
                "model only expects {} inputs".format(
                    len(parameters), len(input_tensors)
                )
            )

        if len(parameters) == 1:
            # if we have simple 1 -> 1 mapping,
            # don't impose requirements on matching
            # names and just use the given input
            # tensor as-is
            input_names = list(input_tensors)
        else:
            input_names = list(parameters)

        try:
            # first try to grab the input tensors
            # by name from the input_tensors
            # dictionary, using the names of the
            # arguments to the model_fn
            input_tensors = [input_tensors[name] for name in input_names]
        except KeyError:
            # at least one of the argument names didn't
            # match an input name in the config, so use
            # the ordering in the config as the backup
            input_tensors = list(input_tensors.values())

        # run the model_fn on the inputs in order
        # to get the model outputs
        outputs = model_fn(*input_tensors)

        # if we only have one output, wrap it
        # in a list to keep things generic
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        # grab the output shapes
        shapes = [tuple(x.shape) for x in outputs]

        # if any of the inputs have a variable length batch
        # dimension, then each output should have a variable
        # length batch dimension too
        if any([x.dims[0] == -1 for x in self.config.input]):
            shapes = [(None,) + s[1:] for s in shapes]

        # if we provided names for the outputs, return them
        # as a dict for validation against the config
        shapes = {f"OUTPUT__{i}": j for i, j in enumerate(shapes)}
        return shapes

    def export(self, model_fn, export_path, verbose=0, **kwargs):
        inputs = []
        for input in self.config.input:
            inputs.append(self._get_tensor(input.dims))

        if len(inputs) == 1:
            inputs = inputs[0]
        else:
            inputs = tuple(inputs)

        # export to a BytesIO object so that we
        # can copy the bytes to cloud filesystems
        export_obj = BytesIO()
        trace = torch.jit.trace(model_fn, inputs)
        torch.jit.save(trace, export_obj)

        # write the written bytes and return
        # the path to which they were written
        self.fs.write(export_obj.getvalue(), export_path)
        return export_path
