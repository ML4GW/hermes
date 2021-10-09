import os
from typing import Optional, Tuple

from hermes.stillwater import PipelineProcess

try:
    from hermes.quiver.io import GCSFileSystem

    _has_gcs = True
except ImportError:
    _has_gcs = False


def _parse_frame_name(fname: str) -> Tuple[int, int]:
    """Use the name of a frame file to infer its initial timestamp and length

    Expects frame names to follow a standard nomenclature
    where the name of the frame file ends {timestamp}-{length}.gwf

    Args:
        fname:
            The name of the frame file
    Returns
        The initial GPS timestamp of the frame file
        The length of the frame file in seconds
    """

    fname = fname.replace(".gwf", "")
    timestamp, length = tuple(map(int, fname.split("-")[-2:]))
    return timestamp, length


class GCSFrameDownloader(PipelineProcess):
    """Process for asynchronously downloading .gwf files from a GCS bucket

    Looks for .gwf files at the specified location in a GCS
    bucket, potentially corresponding to a specified time frame,
    and download them to a local directory. If no .gwf files
    matching the given criteria are met, a `ValueError` is raised.
    The filename of each downloaded file is placed in the output queue.

    Args:
        root:
            The GCS bucket, and potentially the path
            inside of it, at which to look for .gwf files.
            E.g. passing `"ligo-o2"` will look for .gwf files
            anywhere in the bucket `ligo-o2`, while passing
            `"ligo-o2/archive"` will only look for frames
            in `ligo-o2` whose names begin with `archive`
        t0:
            A GPS timestamp. Frames with data entirely before
            this timestamp will be ignored. If `None`, all
            frame files at the location will be downloaded.
        length:
            A length of time in seconds. Frames with data
            entirely after `t0 + length` will be ignored.
            Can't be specified if `t0` is not specified.
            If `None`, all frame files containing data
            after `t0` will be downloaded.
        write_dir:
            A local directory to which to download frame files.
            Will be created if it doesn't already exists. If left
            as `None`, files will be downloaded to the current
            working directory
        credentials:
            Path to a GCP service account JSON credentials file.
            If left as `None`, the value will be inferred from the
            `GOOGLE_APPLICATION_CREDENTIALS` environment variable.
    """

    def __init__(
        self,
        root: str,
        t0: Optional[float] = None,
        length: Optional[float] = None,
        write_dir: Optional[str] = None,
        credentials: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        if not _has_gcs:
            raise RuntimeError(
                "Must install GCS backend to use FrameDownloader"
            )

        # TODO: relax this constraint
        if length is not None and t0 is None:
            raise ValueError(
                "Must specify an initial timestamp t0 "
                "if specifying a length"
            )

        fs = GCSFileSystem(root, credentials)

        # check for gwf frames at either the 1st
        # or 2nd levels of the specified bucket
        blobs = {}
        for blob in fs.bucket.list_blobs(prefix=fs.root):
            if blob.name.endswith(".gwf"):
                timestamp, frame_length = _parse_frame_name(blob.name)

                if t0 is not None and t0 <= timestamp + frame_length:
                    # if we specified a t0 and this timestamp
                    # is greater than it, check to make sure
                    # that we're not outside of any window
                    # specified by length if one was given
                    if length is not None and timestamp > t0 + length:
                        continue
                elif t0 is not None:
                    # otherwise, if we specified a t0 and
                    # this frame is before it, move on
                    continue

                # index them by timestamp so we can sort after the fact
                blobs[timestamp] = blob

        # if we didn't find any blobs, make the error
        # as specific as we can
        if len(blobs) == 0:
            msg = "Couldn't find any frame files {} at location gs://{}"
            if t0 is not None:
                if length is not None:
                    middle = "between timestamps {} and {}".format(
                        t0, t0 + length
                    )
                else:
                    middle = f"after timestamp {t0}"
            else:
                middle = ""

            raise ValueError(msg.format(middle, root))

        # sort the blobs by timestamps
        self.blobs = [blobs[t] for t in sorted(blobs.keys())]

        # create the indicated download directory
        # if doesn't already exist
        if write_dir is not None and not os.path.exists(write_dir):
            os.mkdirs(write_dir)
        self.write_dir = write_dir

        super().__init__(*args, **kwargs)

    def run(self) -> None:
        self.blob_iter = iter(self.blobs)
        super().run()

    def get_package(self) -> str:
        blob = next(self.blob_iter)
        fname = blob.name.split("/")[-1]
        if self.write_dir is not None:
            fname = os.path.join(self.write_dir, fname)

        blob.download_to_filename(fname)
        return fname
