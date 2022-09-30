from typing import TYPE_CHECKING, Optional, Tuple

import torch

from hermes.quiver.streaming import utils as streaming_utils

if TYPE_CHECKING:
    from hermes.quiver import Model, ModelRepository
    from hermes.quiver.model import ExposedTensor


class OnlineAverager(torch.nn.Module):
    def __init__(
        self, update_size: int, batch_size: int, num_updates: int
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
        self.register_buffer(
            "pad",
            torch.zeros(
                update_size * batch_size,
            ),
        )

    def forward(
        self,
        update: torch.Tensor,
        snapshot: torch.Tensor,
        update_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for i in range(self.batch_size):
            x = update[i, -self.snapshot_size :]
            weights = self.normalizer.clamp(self.zero, update_idx + i + 1)

            start = i * self.update_size
            stop = start + x.shape[-1]
            prev = snapshot[start:stop]
            snapshot[start:stop] += (x - prev) / weights

        output_size = self.update_size * self.batch_size
        snapshot_size = snapshot.shape[-1] - output_size
        output, snapshot = torch.split(
            snapshot, [output_size, snapshot_size], dim=-1
        )

        snapshot = torch.concat([snapshot, self.pad])
        return output[None], snapshot, update_idx + self.batch_size


def make_streaming_output_model(
    repository: "ModelRepository",
    input: "ExposedTensor",
    update_size: int,
    batch_size: int,
    num_updates: int,
    name: Optional[str] = None,
    streams_per_gpu: int = 1,
) -> "Model":
    averager = OnlineAverager(update_size, batch_size, num_updates)
    input_batch, kernel_size = input.shape
    if input_batch is not None and input_batch != batch_size:
        raise ValueError(
            "Can't create streaming output model with batch size "
            "of {} from input model with fixed batch size {}".format(
                batch_size, input_batch
            )
        )

    snapshot_size = update_size * (batch_size + num_updates)
    return streaming_utils.add_streaming_model(
        repository,
        averager,
        name=name or "aggregator",
        input_name="update",
        input_shape=(batch_size,) + input.shape[1:],
        state_names=["online_average", "update_index"],
        state_shapes=[(snapshot_size,), (1,)],
        output_names=["stream"],
        streams_per_gpu=streams_per_gpu,
    )
