import os
from typing import Optional

from hermes.stillwater import PipelineProcess

try:
    from hermes.quiver.io import GCSFileSystem

    _has_gcs = True
except ImportError:
    _has_gcs = False


def _parse_blob_fname(fname: str):
    fname = fname.replace(".gwf", "")
    timestamp, length = tuple(map(int, fname.split("-")[-2:]))
    return timestamp, length


class GCSFrameDownloader(PipelineProcess):
    """Process for asynchronously downloading .gwf files from a GCS bucket"""

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
                timestamp, _ = _parse_blob_fname(blob.name)

                if t0 is not None and t0 <= timestamp:
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

        self.write_dir = write_dir
        super().__init__(*args, **kwargs)

    def run(self):
        self.blob_iter = iter(self.blobs)
        super().run()

    def get_package(self):
        blob = next(self.blob_iter)
        fname = blob.name.split("/")[-1]
        if self.write_dir is not None:
            fname = os.path.join(self.write_dir, fname)

        blob.download_to_filename(fname)
        return fname
