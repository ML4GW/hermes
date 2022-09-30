from collections import OrderedDict
from typing import TYPE_CHECKING, Sequence, Tuple

import torch
from tritonclient.grpc import model_config_pb2 as model_config

from hermes.quiver.platform import Platform

if TYPE_CHECKING:
    from hermes.quiver import Model, ModelRepository


def add_streaming_model(
    repository: "ModelRepository",
    streaming_layer: torch.nn.Module,
    name: str,
    input_name: str,
    input_shape: Tuple[int, ...],
    state_names: Sequence[str],
    state_shapes: Sequence[Tuple[int, ...]],
    output_names: Sequence[str],
    streams_per_gpu: int,
) -> "Model":
    model = repository.add(name=name, platform=Platform.ONNX, force=True)
    state_inputs = ["input_" + i for i in state_names]
    state_outputs = ["output_" + i for i in state_names]

    inputs = [(input_name, input_shape)]
    inputs.extend(zip(state_inputs, state_shapes))
    model.export_version(
        streaming_layer,
        input_shapes=OrderedDict(inputs),
        output_names=output_names + state_outputs,
    )

    states = []
    for state_name in state_names:
        state_input = model.config.input.pop(1)
        model.config.output.pop(len(output_names))

        state = model_config.ModelSequenceBatching.State(
            dims=state_input.dims,
            input_name="input_" + state_name,
            output_name="output_" + state_name,
            data_type=state_input.data_type,
            initial_state=[
                model_config.ModelSequenceBatching.InitialState(
                    name=state_name,
                    data_type=state_input.data_type,
                    zero_data=True,
                    dims=state_input.dims,
                )
            ],
        )
        states.append(state)

    sequence_batching = model_config.ModelSequenceBatching(
        max_sequence_idle_microseconds=10000000,
        direct=model_config.ModelSequenceBatching.StrategyDirect(),
        state=states,
    )

    model.config.sequence_batching.MergeFrom(sequence_batching)
    model.config.add_instance_group(count=streams_per_gpu)
    model.config.write()
    return model
