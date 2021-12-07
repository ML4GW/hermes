from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Sequence

import tensorflow as tf

from hermes.quiver.streaming import utils as streaming_utils

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
        if input_shape[-1] > self.snapshot_size:
            raise ValueError(
                "Update size {} cannot be larger than snapshot size {}".format(
                    input_shape[-1], self.snapshot_size
                )
            )

        # snapshot state is maintained like a model
        # weight might be, since this creates a TF
        # Variable which can be assigned to
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
    snapshot_layer = Snapshotter(inputs[0].shape[-1], channels)
    return streaming_utils.add_streaming_model(
        repository,
        snapshot_layer,
        name=name or "snapshotter",
        input_name="stream",
        input_shape=(sum(channels.values()), stream_size),
        streams_per_gpu=streams_per_gpu,
    )
