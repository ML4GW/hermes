from typing import TYPE_CHECKING, Optional, Tuple

import torch

from hermes.quiver.streaming import utils as streaming_utils

if TYPE_CHECKING:
    from hermes.quiver import Model, ModelRepository
    from hermes.quiver.model import ExposedTensor


class OnlineAverager(torch.nn.Module):
    def __init__(
        self,
        update_size: int,
        batch_size: int,
        num_updates: int,
        num_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.update_size = update_size
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.snapshot_size = update_size * num_updates

        normalizer = torch.arange(num_updates) + 1
        normalizer = normalizer.flip(-1)
        normalizer = torch.repeat_interleave(normalizer, update_size)

        self.register_buffer("normalizer", normalizer)
        self.register_buffer("zero", torch.zeros((1,)))

        pad_shape = (update_size * batch_size,)
        if num_channels is not None:
            pad_shape = (num_channels,) + pad_shape
        pad = torch.zeros(pad_shape)
        self.register_buffer("pad", pad)

    def forward(
        self,
        update: torch.Tensor,
        snapshot: torch.Tensor,
        update_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for i in range(self.batch_size):
            if update.ndim > 2:
                x = update[i, :, -self.snapshot_size :]
            else:
                x = update[i, -self.snapshot_size :]

            weights = self.normalizer.clamp(self.zero, update_idx + i + 1)
            start = i * self.update_size
            stop = start + x.shape[-1]

            if update.ndim > 2:
                prev = snapshot[:, start:stop]
                snapshot[:, start:stop] += (x - prev) / weights
            else:
                prev = snapshot[start:stop]
                snapshot[start:stop] += (x - prev) / weights

        output_size = self.update_size * self.batch_size
        snapshot_size = snapshot.shape[-1] - output_size
        output, snapshot = torch.split(
            snapshot, [output_size, snapshot_size], dim=-1
        )

        snapshot = torch.concat([snapshot, self.pad], axis=-1)
        return output[None], snapshot, update_idx + self.batch_size


def make_streaming_output_model(
    repository: "ModelRepository",
    input: "ExposedTensor",
    update_size: int,
    num_updates: int,
    batch_size: Optional[int] = None,
    name: Optional[str] = None,
    streams_per_gpu: int = 1,
) -> "Model":
    if len(input.shape) == 3:
        input_batch, num_channels, kernel_size = input.shape
    elif len(input.shape) == 2:
        input_batch, kernel_size = input.shape
        num_channels = None
    else:
        raise ValueError(
            "Can't produce streaming output state for "
            "tensor with {} dimensions".format(len(input.shape))
        )

    if (num_updates * update_size) > kernel_size:
        raise ValueError(
            "Not enough data for {} updates of length {} "
            "in kernel of length {}".format(
                num_updates, update_size, kernel_size
            )
        )

    if batch_size is None and input_batch is None:
        raise ValueError(
            "Must specify batch size for streaming output "
            "model if corresponding input batch size is variable."
        )
    elif batch_size is None:
        batch_size = input_batch
    elif input_batch is not None and input_batch != batch_size:
        raise ValueError(
            "Can't create streaming output model with batch size "
            "of {} from input model with fixed batch size {}".format(
                batch_size, input_batch
            )
        )

    averager = OnlineAverager(
        update_size, batch_size, num_updates, num_channels
    )

    snapshot_size = update_size * (num_updates + batch_size - 1)
    snapshot_shape = (snapshot_size,)
    if num_channels is not None:
        snapshot_shape = (num_channels,) + snapshot_shape

    return streaming_utils.add_streaming_model(
        repository,
        averager,
        name=name or "aggregator",
        input_name="update",
        input_shape=(batch_size,) + input.shape[1:],
        state_names=["online_average", "update_index"],
        state_shapes=[snapshot_shape, (1,)],
        output_names=["output_stream"],
        streams_per_gpu=streams_per_gpu,
    )
