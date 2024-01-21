from contextlib import ExitStack
from typing import TYPE_CHECKING, Optional, Union

import tensorrt as trt

if TYPE_CHECKING:
    from tritonclient.grpc.model_config_pb2 import ModelConfig


def convert_network(
    model_binary: Union[str, bytes],
    config: "ModelConfig",
    use_fp16: bool = False,
) -> Optional[bytes]:
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
) -> Optional[bytes]:
    """
    using a cheap wrapper to save myself some tabs
    """

    # do some TRT boilerplate initialization
    logger = trt.Logger()
    builder = stack.enter_context(trt.Builder(logger))

    # if the model config doesn't specify a max
    # batch size, the config's value will read 0,
    # so replace with 1 as a default here for streaming
    builder.max_batch_size = max(model_config.max_batch_size, 1)

    # if any of the inputs have a variable
    # length batch dimension, create an
    # optimization profile for that input with
    # the most optimized batch size being the largest
    config = stack.enter_context(builder.create_builder_config())
    config.max_workspace_size = 1 << 28
    if use_fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
        # builder.strict_type_constraints = True

    for input in model_config.input:
        if input.dims[0] != -1:
            # this input doesn't have a variable
            # length batch dimension, so move on
            continue
        elif any([i is None for i in input.dims[1:]]):
            # otherwise if we specified another dim to
            # be variable, we can't support this at the
            # moment so raise an error
            raise ValueError(
                "Can't support variable length dimensions "
                "for any dim other than batch"
            )

        profile = builder.create_optimization_profile()
        min_shape = tuple([1] + input.dims[1:])
        max_shape = tuple([builder.max_batch_size] + input.dims[1:])
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
            "Number of config outputs {} doesn't "
            "match number of outputs {} in network.".format(
                len(model_config.output), network.num_outputs
            )
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
                "Number of dimensions {} specified for "
                "output {} with shape {} not equal to number {} found "
                "in TensorRT network with shape {}".format(
                    len(output.dims),
                    output.name,
                    output.dims,
                    len(network_output.shape),
                    network_output.shape,
                )
            )

        # now iterate through each dimension and make
        # sure that any non-variable length dimensions
        # have the appropriate shape
        for ndim, cdim in zip(network_output.shape, output.dims):
            if ndim != -1 and ndim != cdim:
                raise ValueError(
                    "Shape mismatch for output {} between "
                    "config shape {} and network shape {}".format(
                        output.name, output.dims, network_output.shape
                    )
                )

    # now build the cuda engine and return
    # its serialized binary. Note that if
    # something goes wrong with the build, this
    # won't raise an error, but will rather return
    # None. Allowing this to happen so higher-level
    # converters can make decisions about what to
    # do if/when things go wrong
    return builder.build_serialized_network(network, config)
