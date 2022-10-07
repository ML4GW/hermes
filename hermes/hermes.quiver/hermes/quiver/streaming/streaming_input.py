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
    ) -> Tuple[torch.Tensor, ...]:
        snapshot = snapshot[:, :, self.stride_size :]
        snapshot = torch.cat([snapshot, update], axis=-1)

        if self.batch_size > 1:
            snapshots = snapshot[:, :, None]
            snapshots = self.unfold(snapshots)

            num_channels = sum([i or 1 for i in self.channels_per_snapshot])
            snapshots = snapshots.reshape(num_channels, self.batch_size, -1)
            snapshots = snapshots.transpose(1, 0)
        else:
            snapshots = snapshot

        if len(self.channels_per_snapshot) > 1:
            splits = [i or 1 for i in self.channels_per_snapshot]
            snapshots = torch.split(snapshots, splits, dim=1)

            it = zip(self.channels_per_snapshot, snapshots)
            snapshots = [x if i != 0 else x[:, 0] for i, x in it]
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

    shapes = []
    snapshot_size = None
    for x in inputs:
        if len(x.shape) == 3:
            shape = x.shape
        elif len(x.shape) == 2:
            shape = (x.shape[0], 0, x.shape[1])
        else:
            raise ValueError(
                "Can't make streaming input for tensor {} "
                "with shape {}".format(x.name, x.shape)
            )
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

    update_size = stride_size * batch_size
    if update_size > snapshot_size:
        raise ValueError(
            "Can't use snapshotter with update size {} "
            "greater than snapshot size {}".format(update_size, snapshot_size)
        )

    channels = [i[1] for i in shapes]
    snapshot_layer = Snapshotter(
        snapshot_size, stride_size, batch_size, channels
    )

    num_channels = sum([i or 1 for i in channels])
    return streaming_utils.add_streaming_model(
        repository,
        snapshot_layer,
        name=name or "snapshotter",
        input_name="snapshot_update",
        input_shape=(1, num_channels, update_size),
        state_names=["snapshot"],
        state_shapes=[(1, num_channels, snapshot_size)],
        output_names=[f"{x.model.name}.{x.name}_snapshot" for x in inputs],
        streams_per_gpu=streams_per_gpu,
    )
