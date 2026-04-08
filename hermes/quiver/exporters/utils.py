import inspect
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, Union

from .exporter import Exporter

try:
    import torch

    _has_torch = True
except ImportError:
    _has_torch = False

if TYPE_CHECKING:
    from hermes.quiver.model import Model


def find_exporter(
    model_fn: Union[Callable, "Model"], model: "Model"
) -> Exporter:
    """Find an Exporter capable of exporting the given model function

    Recursively iterates through all sublcasses of the `Exporter`
    class to find the first for which `model_fn` is an instance of
    `Exporter.handles` and for which `model.platform == Exporter.platform`

    Args:
        model_fn:
            The framework-specific function which performs the
            neural network's input/output mapping
        model:
            The `Model` object which specifies the desired export platform
            and configuration information for the model
    Returns:
        An exporter to export `model_fn` to the format specified
        by this models' inference platform
    """

    def _get_all_subclasses(cls):
        """Utility function for recursively finding all Exporters."""
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(_get_all_subclasses(subclass))
        return all_subclasses

    # first try to find an available exporter
    # than can map from the provided model function
    # to the desired inference platform
    for exporter in _get_all_subclasses(Exporter):
        if exporter.platform == model.platform:
            try:
                if isinstance(model_fn, exporter.handles):
                    # if this exporter matches our criteria,
                    # initialize it with this model and return it
                    return exporter(model.config, model.fs)
            except ImportError:
                # evidently we don't have whatever it handles
                # installed, so move on
                continue
    else:
        raise TypeError(
            "No registered exporters which map from "
            "model function type {} to platform {}".format(
                type(model_fn), model.platform
            )
        )


def get_input_names_from_torch_object(
    model_fn: Union["torch.nn.Module", "torch.jit.ScriptModule"],
):
    """
    Parse either a torch.nn.Module or torch.ScriptModule for input names
    """

    if isinstance(model_fn, torch.jit.ScriptModule):
        graph = model_fn.graph
        input_names = [
            node.debugName().split(".")[0] for node in graph.inputs()
        ]
        if "self" in input_names:
            input_names.remove("self")
        return OrderedDict({name: name for name in input_names})
    signature = inspect.signature(model_fn.forward)
    return OrderedDict(signature.parameters)
