from contextlib import ExitStack
from typing import TYPE_CHECKING

import tensorrt as trt

if TYPE_CHECKING:
    from tritonclient.grpc.model_config_pb2 import ModelConfig

# Minimum GPU compute capability (SM × 10) supported by TRT, keyed by the
# (major, minor) version from which each applies. TRT 10.0 deprecated Volta
# (SM 7.0) and 10.5 removed it.
_TRT_MIN_SM = {(8, 0): 35, (9, 0): 53, (10, 0): 70, (10, 5): 75}


def _min_sm(trt_version: str) -> int | None:
    """The minimum compute capability TRT `trt_version` supports, if known."""
    version = tuple(int(i) for i in trt_version.split(".")[:2])
    mins = [sm for since, sm in _TRT_MIN_SM.items() if since <= version]
    return max(mins, default=None)


def _check_gpu_trt_compatibility() -> None:
    """Raise a clear RuntimeError if the current GPU is not supported."""
    try:
        import torch
    except ImportError:
        return  # can't check without torch; TRT will surface its own error

    if not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor

    min_sm = _min_sm(trt.__version__)
    if min_sm is not None and sm < min_sm:
        device = torch.cuda.get_device_name()
        min_major, min_minor = divmod(min_sm, 10)
        raise RuntimeError(
            f"GPU '{device}' has compute capability SM {major}.{minor}, "
            f"which is not supported by TensorRT {trt.__version__}. "
            f"It requires SM {min_major}.{min_minor} or later. For "
            "Volta-class GPUs (V100), use TensorRT 10.4 or earlier, and a "
            "Triton container with the same version (24.09 or earlier)."
        )


def convert_network(
    model_binary: str | bytes,
    config: "ModelConfig",
    use_fp16: bool = False,
) -> bytes | None:
    if isinstance(model_binary, str):
        with open(model_binary, "rb") as f:
            model_binary = f.read()

    with ExitStack() as stack:
        return _convert_network(stack, model_binary, config, use_fp16)


def _convert_network(
    stack: ExitStack,
    model_binary: bytes,
    model_config: "ModelConfig",
    use_fp16: bool,
) -> bytes | None:
    """
    using a cheap wrapper to save myself some tabs
    """
    _check_gpu_trt_compatibility()

    # do some TRT boilerplate initialization
    logger = trt.Logger()
    builder = stack.enter_context(trt.Builder(logger))

    # if the model config doesn't specify a max
    # batch size, the config's value will read 0,
    # so replace with 1 as a default here for streaming.
    max_batch_size = max(model_config.max_batch_size, 1)

    # if any of the inputs have a variable
    # length batch dimension, create an
    # optimization profile for that input with
    # the most optimized batch size being the largest
    config = stack.enter_context(builder.create_builder_config())
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    if use_fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
        # builder.strict_type_constraints = True

    for input in model_config.input:
        if input.dims[0] != -1:
            # this input doesn't have a variable
            # length batch dimension, so move on
            continue
        elif any(i is None for i in input.dims[1:]):
            # otherwise if we specified another dim to
            # be variable, we can't support this at the
            # moment so raise an error
            raise ValueError(
                "Can't support variable length dimensions "
                "for any dim other than batch"
            )

        profile = builder.create_optimization_profile()
        min_shape = tuple([1] + input.dims[1:])
        max_shape = tuple([max_batch_size] + input.dims[1:])
        optimal_shape = max_shape

        profile.set_shape(input.name, min_shape, optimal_shape, max_shape)
        config.add_optimization_profile(profile)

    # now create a network to populate with
    # the contents of the ONNX binary
    network = stack.enter_context(
        builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
    )
    parser = stack.enter_context(trt.OnnxParser(network, logger))
    success = parser.parse(model_binary)
    if not success:
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        errors = "\n".join(map(str, errors))
        completed_layers = [
            network.get_layer(i).name for i in range(network.num_layers)
        ]
        msg = "Parsing ONNX binary failed. Completed layers:\n"
        msg += "\n".join(completed_layers)
        msg += "\n\nRaised errors:\n" + errors
        raise RuntimeError(msg)

    if len(model_config.output) == 1 and network.num_outputs == 0:
        # if we only have a single output and for whatever
        # reason the network failed to mark an output layer
        # on itself, just mark the last layer as the output
        last_layer = network.get_layer(network.num_layers - 1)
        network_output = last_layer.get_output(0)
        network.mark_output(network_output)
    elif len(model_config.output) != network.num_outputs:
        # otherwise if the config specifies multiple
        # outputs and/or the number of outputs marked
        # by the ONNX parser doesn't match the number
        # in the config, we don't know how to reconcile
        # this so raise an error
        raise ValueError(
            f"Number of config outputs {len(model_config.output)} doesn't "
            f"match number of outputs {network.num_outputs} in network."
        )

    for n, output in enumerate(model_config.output):
        # assume that network outputs fall in the
        # same order as as they are ordered in
        # the model config. Assign them the correct name
        network_output = network.get_output(n)
        network_output.name = output.name

        # make sure that the number of dimensions for
        # each output match what we would expect
        if len(network_output.shape) != len(output.dims):
            raise ValueError(
                f"Number of dimensions {len(output.dims)} specified for "
                f"output {output.name} with shape {output.dims} not equal "
                f"to number {len(network_output.shape)} found "
                f"in TensorRT network with shape {network_output.shape}"
            )

        # now iterate through each dimension and make
        # sure that any non-variable length dimensions
        # have the appropriate shape
        for ndim, cdim in zip(network_output.shape, output.dims, strict=True):
            if ndim != -1 and ndim != cdim:
                raise ValueError(
                    f"Shape mismatch for output {output.name} between "
                    f"config shape {output.dims} and network shape "
                    f"{network_output.shape}"
                )

    # now build the cuda engine and return
    # its serialized binary. Note that if
    # something goes wrong with the build, this
    # won't raise an error, but will rather return
    # None. Allowing this to happen so higher-level
    # converters can make decisions about what to
    # do if/when things go wrong
    return builder.build_serialized_network(network, config)
