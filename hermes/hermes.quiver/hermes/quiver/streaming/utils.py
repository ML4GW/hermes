from typing import TYPE_CHECKING, Tuple

import tensorflow as tf
from tritonclient.grpc import model_config_pb2 as model_config

from hermes.quiver.platform import Platform

if TYPE_CHECKING:
    from hermes.quiver import Model, ModelRepository


def build_streaming_model(
    input_name: str,
    input_shape: Tuple[int, ...],
    streaming_layer: tf.keras.layers.Layer,
) -> tf.keras.Model:
    input = tf.keras.Input(
        name=input_name, shape=input_shape, batch_size=1, dtype=tf.float32
    )

    # include an input which is used to indicate
    # whether a new sequence is beginning to
    # clear the state
    sequence_start = tf.keras.Input(
        name="sequence_start",
        type_spec=tf.TensorSpec(
            shape=(1,), name="sequence_start"  # TODO: batch size
        ),
    )

    output = streaming_layer(input, sequence_start)

    return tf.keras.Model(inputs=[input, sequence_start], outputs=output)


def add_sequence_config(
    model: "Model", input_name: str, shape: Tuple[int, ...]
) -> None:
    # make the snapshotter model stateful by
    # setting up sequence batching with a control
    # flag for indicating the start of a new sequence
    start = model_config.ModelSequenceBatching.Control.CONTROL_SEQUENCE_START
    model.config.sequence_batching.MergeFrom(
        model_config.ModelSequenceBatching(
            max_sequence_idle_microseconds=10000000,
            direct=model_config.ModelSequenceBatching.StrategyDirect(),
            control_input=[
                model_config.ModelSequenceBatching.ControlInput(
                    name="sequence_start",
                    control=[
                        model_config.ModelSequenceBatching.Control(
                            kind=start,
                            fp32_false_true=[0, 1],
                        )
                    ],
                )
            ],
        )
    )

    # add a model warm up since the first couple
    # inference executions in TensorFlow can be slower
    model.config.model_warmup.append(
        model_config.ModelWarmup(
            inputs={
                input_name: model_config.ModelWarmup.Input(
                    dims=shape,
                    data_type=model_config.TYPE_FP32,
                    zero_data=True,
                )
            },
            name="zeros_warmup",
        )
    )


def export_streaming_model(
    model: "Model", streamer: tf.keras.Model, streams_per_gpu: int
) -> "Model":
    input_layer = streamer.layers[0]
    add_sequence_config(model, input_layer.name, input_layer.output_shape[0])

    # set the number of streams desired per GPU
    model.config.add_instance_group(count=streams_per_gpu)
    model.export_version(streamer)

    # export_version will create an input for `sequence_start`,
    # which we don't want, so delete the input then rewrite
    # the config to the model repository
    del model.config.input[1]
    model.config.write()

    return model


def add_streaming_model(
    repository: "ModelRepository",
    streaming_layer: tf.keras.layers.Layer,
    name: str,
    input_name: str,
    input_shape: Tuple[int, ...],
    streams_per_gpu: int,
) -> "Model":
    streamer = build_streaming_model(input_name, input_shape, streaming_layer)
    model = repository.add(name=name, platform=Platform.SAVEDMODEL, force=True)
    return export_streaming_model(model, streamer, streams_per_gpu)
