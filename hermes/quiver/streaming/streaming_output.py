from typing import TYPE_CHECKING, Optional, Tuple

import torch

from hermes.quiver.streaming import utils as streaming_utils

if TYPE_CHECKING:
    from hermes.quiver import Model, ModelRepository
    from hermes.quiver.model import ExposedTensor


def window(x: torch.Tensor, num_windows: int, stride: int):
    if x.ndim == 2:
        num_channels = len(x)
        x = x.view(1, num_channels, 1, -1)
    else:
        x = x.view(1, 1, 1, -1)
        num_channels = 1

    x = torch.nn.functional.unfold(
        x, kernel_size=(1, num_windows), dilation=(1, stride)
    )
    x = x.reshape(num_channels, num_windows, -1)
    return x.transpose(1, 0)


class OnlineAverager(torch.nn.Module):
    """
    Module for performing stateful online averaging of
    batches of overlapping timeseries. At present, the
    first `num_updates` predictions produced by this
    model will underestimate the true average.

    Args:
        update_size:
            The number of samples separating the timestamps
            of subsequent inputs.
        batch_size:
            The number of batched inputs to expect at inference
            time.
        num_updates:
            The number of steps over which to average predictions
            before returning them.
        num_channels:
            The expected channel dimension of the input passed
            to the module at inference time.
    """

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
        self.num_channels = num_channels
        self.pad = (
            0,
            update_size * batch_size,
        )

        # build a blank tensor into which we will embed
        # the updated snapshot predictions at the
        # appropriate time offset for in-batch averaging
        snapshot_size = (batch_size + num_updates - 1) * update_size
        if num_channels is None:
            blank = torch.zeros((batch_size, snapshot_size))
        else:
            blank = torch.zeros((batch_size, num_channels, snapshot_size))
        self.register_buffer("blank", blank)

        # set up the indices at which the updated snapshots
        # will be embedded into the blank tensor
        idx = torch.arange(num_updates * update_size)
        idx = torch.stack([idx + i * update_size for i in range(batch_size)])
        if num_channels is not None:
            idx = idx.view(batch_size, 1, -1).repeat(1, num_channels, 1)
        self.register_buffer("idx", idx)

        # normalization indices used to downweight the
        # existing average at each in-batch aggregation
        weights = torch.scatter(blank, -1, idx, 1).sum(0)
        weights = window(weights, batch_size, update_size)
        if num_channels is None:
            weights = weights[:, 0]
        self.register_buffer("weights", weights)

    def forward(
        self, update: torch.Tensor, snapshot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # to keep things general, add a dummy dimension
        # to the update if it has no channel dimension
        if self.num_channels is None:
            update = update[:, None]

        # slice off the steps from this update closest
        # to the future that we'll actually use
        keep = self.num_updates * self.update_size
        x = update[:, :, -keep:] / self.num_updates

        # window the existing snapshot into overlapping
        # segments and average them with our new updates
        windowed = window(snapshot, self.batch_size, self.update_size)
        if self.num_channels is None:
            windowed = windowed[:, 0]
            x = x[:, 0]
        windowed /= self.weights
        windowed += x

        # embed these windowed averages into a blank
        # array with offsets so that we can add the
        # overlapping bits
        padded = torch.scatter(self.blank, -1, self.idx, windowed)
        new_snapshot = padded.sum(axis=0)

        if self.num_updates == 1:
            # if we don't need stateful behavior,
            # just return the "snapshot" as-is
            output, new_snapshot = new_snapshot, torch.zeros_like(snapshot)
        else:
            # otherwise split off the values that have finished
            # averaging and are being returned from the ones that
            # will comprise the snapshot at the next update
            splits = [self.batch_size, self.num_updates - 1]
            output, new_snapshot = torch.split(
                new_snapshot, [i * self.update_size for i in splits], dim=-1
            )
            new_snapshot = torch.nn.functional.pad(new_snapshot, self.pad)

        # make sure to add a dummy batch dimension
        # to the output for triton
        return output[None], new_snapshot


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
        state_names=["online_average"],
        state_shapes=[snapshot_shape],
        output_names=["output_stream"],
        streams_per_gpu=streams_per_gpu,
    )
