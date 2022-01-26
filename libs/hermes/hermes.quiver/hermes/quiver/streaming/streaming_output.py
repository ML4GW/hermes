from typing import TYPE_CHECKING, Optional

import numpy as np
import tensorflow as tf

from hermes.quiver.streaming import utils as streaming_utils

if TYPE_CHECKING:
    from hermes.quiver import Model, ModelRepository
    from hermes.quiver.model import ExposedTensor


@tf.keras.utils.register_keras_serializable(name="Aggregator")
class Aggregator(tf.keras.layers.Layer):
    def __init__(
        self, update_size: int, num_updates: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.update_size = update_size
        self.num_updates = num_updates
        self.snapshot_size = update_size * num_updates

    def build(self, input_shape) -> None:
        if input_shape[0] is None:
            raise ValueError("Must specify batch dimension")
        if input_shape[0] != 1:
            # TODO: support batching
            raise ValueError("Batching not currently supported")
        if input_shape[-1] < self.snapshot_size:
            raise ValueError(
                "Expected input update of at least {} samples, but "
                "found {}".format(self.snapshot_size, input_shape[-1])
            )

        self.update_idx = self.add_weight(
            name="update_idx", shape=[], dtype=tf.float32, initializer="zeros"
        )

        snapshot_shape = [
            input_shape[0],
            self.snapshot_size - self.update_size,
        ]
        if len(input_shape) == 3:
            snapshot_shape.insert(1, input_shape[1])
        elif len(input_shape) > 3:
            raise ValueError(
                "Unsupported number of input dimensions {}".format(
                    len(input_shape)
                )
            )

        self.snapshot = self.add_weight(
            name="snapshot",
            shape=snapshot_shape,
            dtype=tf.float32,
            initializer="zeros",
        )

        update_shape = [input_shape[0], self.update_size]
        if len(input_shape) == 3:
            update_shape.insert(1, input_shape[1])

        self.update = tf.zeros(update_shape, dtype=tf.float32)
        self.normalizer = tf.constant(
            np.repeat(np.arange(self.num_updates), self.update_size)[::-1] + 1,
            dtype=tf.float32,
        )

    def call(self, x, sequence_start):
        snapshot = (1.0 - sequence_start) * self.snapshot
        update_idx = (1.0 - sequence_start) * self.update_idx + 1

        if len(x.shape) == 3:
            x = x[:, :, -self.snapshot_size :]
        else:
            x = x[:, -self.snapshot_size]

        snapshot = tf.concat([snapshot, self.update], axis=-1)
        weights = tf.clip_by_value(self.normalizer, 0, update_idx)
        snapshot += (x - snapshot) / weights

        output, snapshot = tf.split(
            snapshot,
            [self.update_size, self.update_size * (self.num_updates - 1)],
            axis=-1,
        )

        self.snapshot.assign(snapshot)
        self.update_idx.assign(update_idx[0])
        return output


def make_streaming_output_model(
    repository: "ModelRepository",
    input: "ExposedTensor",
    update_size: int,
    num_updates: int,
    name: Optional[str] = None,
    streams_per_gpu: int = 1,
) -> "Model":
    aggregator_layer = Aggregator(update_size, num_updates)
    return streaming_utils.add_streaming_model(
        repository,
        aggregator_layer,
        name=name or "aggregator",
        input_name="update",
        input_shape=input.shape[1:],
        streams_per_gpu=streams_per_gpu,
    )
