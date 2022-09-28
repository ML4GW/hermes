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
        self.snapshot_size = update_size * batch_size * num_updates

        normalizer = torch.arange(num_updates * batch_size)
        normalizer = normalizer.repeat(update_size)
        self.register_buffer("normalizer", normalizer)

    def forward(
        self,
        update: torch.Tensor,
        snapshot: torch.Tensor,
        update_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = self.normalizer.clamp(0, update_idx + 1)
        for i in range(self.batch_size):
            start = i * self.update_size
            stop = start + update.shape[-1]

            x = update[i]
            weights = self.normalizer[start:stop]
            snapshot[start:stop] += (x - snapshot[start:stop]) / weights

        output_size = self.update_size * self.batch_size
        snapshot_size = snapshot.shape[-1] - output_size
        output, snapshot = torch.split(
            snapshot, [output_size, snapshot_size], dim=-1
        )

        snapshot = torch.nn.functional.pad(snapshot, (0, output_size))
        return output[None], snapshot, update_idx + 1


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
    if input_batch > 0 and input_batch != batch_size:
        raise ValueError(
            "Can't create streaming output model with batch size "
            "of {} from input model with fixed batch size {}".format(
                batch_size, input_batch
            )
        )

    snapshot_size = update_size * batch_size * num_updates + kernel_size
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
