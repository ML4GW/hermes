from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Sequence

import tensorflow as tf
from tritonclient.grpc import model_config_pb2 as model_config

from hermes.quiver.platform import Platform

if TYPE_CHECKING:
    from hermes.quiver import Model, ModelRepository
    from hermes.quiver.model import ExposedTensor


@tf.keras.utils.register_keras_serializable(name="Snapshotter")
class Snapshotter(tf.keras.layers.Layer):
    """Layer for capturing snapshots of streaming time series

    Args:
        snapshot_size:
            The size of the snapshot to be updated
        channels:
            An ordered dictionary mapping the names of models
            with snapshots to be updated to the number of
            channels in each model
    """

    def __init__(
        self,
        snapshot_size: int,
        channels: OrderedDict,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.snapshot_size = snapshot_size
        self.channels = channels

    def build(self, input_shape) -> None:
        num_channels = sum(self.channels.values())
        if input_shape[0] is None:
            raise ValueError("Must specify batch dimension")
        if input_shape[0] != 1:
            # TODO: support batching
            raise ValueError("Batching not currently supported")
        if num_channels != input_shape[1]:
            raise ValueError(
                "Number of channels specified {} doesn't "
                "match number of channels found {}".format(
                    num_channels, input_shape[1]
                )
            )

        self.snapshot = self.add_weight(
            name="snapshot",
            shape=(input_shape[0], input_shape[1], self.snapshot_size),
            dtype=tf.float32,
            initializer="zeros",
            trainable=False,
        )
        self.update_size = input_shape[2]

    def call(self, stream, sequence_start):
        # grab the non-stale part of the existing snapshot
        # multiply it by 0 if the sequence has been restarted
        old = (1.0 - sequence_start) * self.snapshot[:, :, self.update_size :]

        # create a new snapshot using the update and
        # assign it to the variable
        update = tf.concat([old, stream], axis=2)
        self.snapshot.assign(update)

        # split the updated snapshot into the various
        # channels required for each model
        return tf.split(update, list(self.channels.values()), axis=1)

    def compute_output_shape(self, input_shapes):
        return [
            tf.TensorShape([input_shapes[0][0], i, self.snapshot_size])
            for i in self.channels.values()
        ]

    def get_config(self):
        config = {
            "snapshot_size": self.snapshot_size,
            "channels": self.channels,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def make_streaming_input_model(
    repository: "ModelRepository",
    inputs: Sequence["ExposedTensor"],
    stream_size: int,
    name: Optional[str] = None,
    streams_per_gpu: int = 1,
) -> "Model":
    """Create a snapshotter model and add it to the repository"""

    if len(inputs) > 1 and not all(
        [x.shape[-1] == inputs[0].shape[-1] for x in inputs]
    ):
        raise ValueError(
            "Cannot create streaming inputs for inputs "
            "with shapes {}".format([x.shape[-1] for x in inputs])
        )

    # TODO: support 2D streaming
    channels = OrderedDict([(x.model.name, x.shape[1]) for x in inputs])

    # construct the inputs to the model
    input = tf.keras.Input(
        name="stream",
        shape=(sum(channels.values()), stream_size),
        batch_size=1,  # TODO: other batch sizes
        dtype=tf.float32,
    )
    sequence_start = tf.keras.Input(
        name="sequence_start",
        type_spec=tf.TensorSpec(
            shape=(1,), name="sequence_start"  # TODO: batch size
        ),
    )

    # construct and call the snapshotter layer
    snapshot_layer = Snapshotter(inputs[0].shape[-1], channels)
    output = snapshot_layer(input, sequence_start)

    # build the model
    inputs = [input, sequence_start]
    snapshotter = tf.keras.Model(inputs=inputs, outputs=output)

    model = repository.add(
        name=name or "snapshotter", platform=Platform.SAVEDMODEL, force=True
    )

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

    model.config.model_warmup.append(
        model_config.ModelWarmup(
            inputs={
                "stream": model_config.ModelWarmup.Input(
                    dims=[1, sum(channels.values()), stream_size],
                    data_type=model_config.TYPE_FP32,
                    zero_data=True,
                )
            },
            name="zeros_warmup",
        )
    )
    model.config.add_instance_group(count=streams_per_gpu)
    model.export_version(snapshotter)

    del model.config.input[1]
    model.config.write()

    return model
