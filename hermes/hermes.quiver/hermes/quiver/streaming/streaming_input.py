from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch

from hermes.quiver.streaming import utils as streaming_utils

if TYPE_CHECKING:
    from hermes.quiver import Model, ModelRepository
    from hermes.quiver.model import ExposedTensor


class Snapshotter(torch.nn.Module):
    def __init__(
        self,
        snapshot_size: int,
        stride_size: int,
        batch_size: int,
        channels_per_snapshot: Sequence[int],
    ) -> None:
        super().__init__()

        self.snapshot_size = snapshot_size
        self.stride_size = stride_size
        self.batch_size = batch_size
        self.channels_per_snapshot = list(channels_per_snapshot)

        if batch_size > 1:
            self.unfold = torch.nn.Unfold(
                (1, batch_size), dilation=(1, stride_size)
            )
        else:
            self.unfold = None

    def forward(
        self, update: torch.Tensor, snapshot: torch.Tensor
    ) -> Tuple[torch.tensor, ...]:
        snapshot = snapshot[:, :, self.stride_size :]
        snapshot = torch.cat([snapshot, update], axis=-1)

        if self.batch_size > 1:
            snapshots = snapshot[:, :, None]
            snapshots = self.unfold(snapshots)
            snapshots = snapshots.reshape(
                sum(self.channels_per_snapshot), self.batch_size, -1
            )
            snapshots = snapshots.transpose(1, 0)
        else:
            snapshots = snapshot

        if len(self.channels_per_snapshot) > 1:
            snapshots = torch.split(
                snapshots, self.channels_per_snapshot, dim=1
            )
        else:
            snapshots = (snapshots,)

        snapshot = snapshot[:, :, -self.snapshot_size :]
        return tuple(snapshots) + (snapshot,)


def make_streaming_input_model(
    repository: "ModelRepository",
    inputs: Sequence["ExposedTensor"],
    stride_size: int,
    batch_size: int = 1,
    name: Optional[str] = None,
    streams_per_gpu: int = 1,
) -> "Model":
    """Create a snapshotter model and add it to the repository"""

    shapes, snapshot_size, = (
        [],
        None,
    )
    for x in inputs:
        if len(x.shape) > 3:
            raise ValueError(
                "Can't make streaming input for tensor {} "
                "with shape {}".format(x.name, x.shape)
            )
        shape = x.shape if len(x) == 3 else (x.shape[0], 1, x.shape[1])
        shapes.append(shape)

        if snapshot_size is not None and shape[-1] != snapshot_size:
            raise ValueError(
                "Input for tensor {} has last dimension {} "
                "which doesn't match expected last dimension {} "
                "from tensor {}".format(
                    x.name, shape[-1], snapshot_size, inputs[0].name
                )
            )
        elif snapshot_size is None:
            snapshot_size = shape[-1]

    channels = [x.shape[1] for x in shapes]
    snapshot_layer = Snapshotter(
        snapshot_size, stride_size, batch_size, channels
    )

    update_size = stride_size * batch_size
    return streaming_utils.add_streaming_model(
        repository,
        snapshot_layer,
        name=name or "snapshotter",
        input_name="stream",
        input_shape=(sum(channels), update_size),
        state_name="snapshot",
        state_shape=(sum(channels), snapshot_size + update_size),
        output_names=[x.name + "_snapshot" for x in inputs],
        streams_per_gpu=streams_per_gpu,
    )
