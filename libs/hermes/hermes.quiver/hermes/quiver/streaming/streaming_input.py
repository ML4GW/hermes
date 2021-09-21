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

    def call(self, stream):
        update = tf.concat(
            [self.snapshot[:, :, self.update_size :], stream], axis=2
        )
        self.snapshot.assign(update)
        splits = tf.split(update, list(self.channels.values()), axis=1)

        outputs = []
        for output, name in zip(splits, self.channels):
            outputs.append(tf.identity(output, name=name))
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape([input_shape[0], i, self.snapshot_size])
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
    if len(inputs) > 1 and not all(
        [x.shape[-1] == inputs[0].shape[-1] for x in inputs]
    ):
        raise ValueError(
            "Cannot create streaming inputs for inputs "
            "with shapes {}".format([x.shape[-1] for x in inputs])
        )

    # TODO: support 2D streaming
    channels = OrderedDict([(x.model.name, x.shape[1]) for x in inputs])
    input = tf.keras.Input(
        name="stream",
        shape=(sum(channels.values()), stream_size),
        batch_size=1,  # TODO: other batch sizes
        dtype=tf.float32,
    )
    snapshot_layer = Snapshotter(inputs[0].shape[-1], channels)
    output = snapshot_layer(input)
    snapshotter = tf.keras.Model(inputs=input, outputs=output)

    model = repository.add(
        name=name or "snapshotter", platform=Platform.SAVEDMODEL, force=True
    )
    model.config.sequence_batching = model_config.ModelSequenceBatching(
        max_sequence_idle_microseconds=10000000,
        direct=model_config.ModelSequenceBatching.StrategyDirect(),
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

    return model
