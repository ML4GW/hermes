import os
import sys
import time
from multiprocessing import Queue
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from hermes.stillwater import Package, PipelineProcess

try:
    from hermes.quiver.io import GCSFileSystem

    _has_gcs = True
except ImportError:
    _has_gcs = False


try:
    from gwpy.timeseries import TimeSeriesDict

    _has_gwpy = True
except ImportError:
    _has_gwpy = False


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

                if t0 is not None and t0 < timestamp + frame_length:
                    # if we specified a t0 and this timestamp
                    # is greater than it, check to make sure
                    # that we're not outside of any window
                    # specified by length if one was given
                    if length is not None and timestamp >= t0 + length:
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
            os.makedirs(write_dir)
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


class FrameCrawler(PipelineProcess):
    """Waits for data to become available in the designated directory

    Simple process which can be used for LIGO data replay streams
    to wait for new frames to be written to a specified directory.
    Infers the file pattern by replacing the timestamp with `"{}"`,
    and starts with the latest timestamp in the directory. Subsequent
    timestamps are inferred from the length of the current frame,
    which is inferred from the filename as well.

    Args:
        data_dir:
            The directory to look for frames in
        timeout:
            The amount of time to wait without finding a new
            frame before a `RuntimeError` is raised. If left
            as `None`, the process will wait indefinitely
    """

    def __init__(
        self,
        data_dir: str,
        timeout: Optional[float] = None,
        N: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.timeout = timeout
        self.N = N

    def run(self):
        # override the parent run briefly to figure
        # out how to iterate through the designated
        # data directory
        try:
            # iterate through all the filenames in the
            # directory and parse out the timestamps
            # for all frame files
            self.timestamp, self.pattern = None, None
            for fname in os.listdir(self.data_dir):
                # ignore any non frame files, though these
                # really shouldn't be in here either
                if not fname.endswith(".gwf"):
                    continue

                # parse the timestamp out of the file
                # and use it to build a pattern that
                # we can fill in with future timestamps
                # save the length as an attribute for
                # iteration later
                tstamp, self.length = _parse_frame_name(fname)
                subbed = fname.replace(str(tstamp), "{}")

                # make sure that all frame files adhere
                # to the same pattern, otherwise our job
                # becomes really difficult
                if self.pattern is None:
                    self.pattern = subbed
                else:
                    if self.pattern != subbed:
                        raise ValueError(
                            "Found inconsistent file patterns "
                            "in data dir {}: {} and {}".format(
                                self.data_dir, self.pattern, subbed
                            )
                        )

                # find the latest timestamp to begin with
                if self.timestamp is None or tstamp > self.timestamp:
                    self.timestamp = tstamp

            # if we never found any .gwf files, raise an error
            if self.timestamp is None:
                raise ValueError(
                    "Couldn't find any .gwf files in data directory {}".format(
                        self.data_dir
                    )
                )

            # keep track of when the last file became available
            # for measuring timeouts if we specified one
            self._last_time = time.time()
        except Exception as e:
            self.cleanup(e)
            sys.exit(1)

        # run the normal process target
        self.n = 0
        super().run()

    def get_package(self):
        if self.n == self.N:
            raise StopIteration

        fname = os.path.join(
            self.data_dir, self.pattern.format(self.timestamp)
        )

        # wait for the current file to exist, possibly
        # timing out if it takes too long
        while not os.path.exists(fname):
            if (
                self.timeout is not None
                and (time.time() - self._last_time) > self.timeout
            ):
                raise RuntimeError(
                    "Couldn't find frame file named {} "
                    "after {} seconds".format(fname, self.timeout)
                )
            time.sleep(1e-6)

        # reset our timeout counter and advance the
        # timestamp by the length of the frames
        self._last_time = time.time()
        self.timestamp += self.length

        # return the formatted filename for passing
        # to downstream loader processes
        self.n += 1
        return fname


class FrameLoader(PipelineProcess):
    """Loads .gwf frames using filenames passed from upstream processes

    Loads gwf frames as gwpy TimeSeries dicts, then resamples
    and converts these frames to numpy arrays according to the
    order specified in `channels`. If strain data is present,
    it's broken off before preprocessing is applied, and the
    loaded data is then iterated through in (possibly overlapping)
    chunks.

    Args:
        chunk_size:
            The size of chunk to return at each iteration, in samples
        step_size:
            The number of samples to take between returned chunks
        sample_rate:
            The rate at which to resample the loaded data
        channels:
            The channels from the frame to load. If `strain_q`
            is not `None`, and therefore there is strain data
            that ought to be processed separately, it will be
            assumed to be the first channel in this sequence.
        t0:
            An initial timestamp before which to ignore data.
            Useful if you only want to process fractions of
            the first and last frames in order to distribute
            workloads evenly among many clients. If left as
            `None`, all frames will be processed in full.
        length:
            The amount of time from `t0` to process data, in
            seconds. Ignored if `t0` is `None`, and if left
            as `None` all frames after the first will be
            processed in full.
        sequence_id:
            A sequence id to assign to this stream of data.
            Unnecessary if not performing stateful streaming
            inference.
        preprocessor:
            A preprocessing function which will be applied
            to the data at load time after resampling. If
            `strain_q` is not `None`, this will _not_ be
            applied to the first channel in `channels`.
        strain_q:
            A queue in which to put the loaded filenames
            as well as the first channel of the resampled
            data. Useful if the first channel represents
            strain data that you want to process separately
            from the other channels in the stream or avoid
            sending to the client.
        remove:
            Whether to remove frames from disk once they're read
    """

    def __init__(
        self,
        chunk_size: int,
        step_size: int,
        sample_rate: float,
        channels: Sequence[str],
        t0: Optional[float] = None,
        length: Optional[float] = None,
        sequence_id: Optional[int] = None,
        preprocessor: Optional[Callable] = None,
        strain_q: Optional[Queue] = None,
        remove: bool = False,
        *args,
        **kwargs,
    ) -> None:
        if not _has_gwpy:
            raise RuntimeError("Must install gwpy to use FrameLoader")
        super().__init__(*args, **kwargs)

        self.chunk_size = chunk_size
        self.step_size = step_size
        self.sample_rate = sample_rate
        self.sequence_id = sequence_id
        self.channels = channels
        self.t0 = t0
        self.length = length
        self.preprocessor = preprocessor
        self.strain_q = strain_q
        self.remove = remove

        self._idx = 0
        self._frame_idx = 0
        self._data = None
        self._end_next = False

    def load_frame(self) -> np.ndarray:
        # get the name of the next file to load from
        # an upstream process, possibly raising a
        # StopIteration if that process is done
        fname = super().get_package()

        # load in the data and prepare it as a numpy array
        data = TimeSeriesDict.read(fname, self.channels)
        data.resample(self.sample_rate)
        data = np.stack(
            [data[channel].value for channel in self.channels]
        ).astype("float32")

        # if we specified explicit time boundaries on the
        # data that we wanted to process, use the filename
        # to infer which parts of the data that we want
        # to slice off if we're at one of the boundaries
        if self.t0 is not None and self._data is None:
            # only need to look for data that's too early
            # if we haven't processed any data yet, since
            # we're going to assume that frames are being
            # passed in chronological order for this use case
            tstamp, _ = _parse_frame_name(fname)
            diff = self.t0 - tstamp
            idx = int(diff * self.sample_rate)
            data = data[:, idx:]
        elif self.length is not None:
            # however, we need to constantly looking at
            # whether we're about to go over time if we
            # process the entire frame's worth of data
            tstamp, frame_length = _parse_frame_name(fname)
            if (tstamp + frame_length) > (self.t0 + self.length):
                diff = tstamp + frame_length - self.t0 - self.length
                idx = int(diff * self.sample_rate)
                data = data[:, :-idx]

        # if we want to handle strain data separately,
        # ship it in a queue before we do the preprocessing
        if self.strain_q is not None:
            strain, data = data[0], data[1:]
            self.strain_q.put((fname, strain))

        # apply preprocessing to the remaining channels
        if self.preprocessor is not None:
            data = self.preprocessor(data)

        # remove the file now that we're done with it
        # if we indicated to do so
        if self.remove:
            os.remove(fname)

        return data

    def get_package(self) -> Package:
        start = self._frame_idx * self.step_size
        stop = start + self.chunk_size

        # look ahead at whether we'll need to grab a frame
        # after this timestep so that we can see if this
        # will be the last step that we take
        # if sequence_end is False and therefore self._end_next
        # is False, we don't need to try to get another frame
        # since this will be the last one
        next_stop = stop + self.step_size
        sequence_start, sequence_end = False, self._end_next
        if (
            self._data is None or next_stop >= self._data.shape[1]
        ) and not sequence_end:
            if self._data is None:
                sequence_start = True

            # try to load in the next frame's worth of data
            try:
                data = self.load_frame()
            except StopIteration:
                # super().get_package() raised a StopIteration,
                # so catch it and indicate that this will be
                # the last inference that we'll produce
                if next_stop == self._data.shape[1]:
                    # if the next frame will end precisely at the
                    # end of our existing data, we'll have one more
                    # frame to process after this one, so set
                    # self._end_next to True
                    self._end_next = True
                else:
                    # otherwise, the next frame wouldn't be able
                    # to fit into the model input, so we're going
                    # to end here and just accept that we'll have
                    # some trailing data.
                    # TODO: should we append with zeros? How will
                    # this information get passed to whatever process
                    # is piecing information together at the other end?
                    sequence_end = True
            else:
                # otherwise append the new data to whatever
                # remaining data we have left to go through
                if self._data is not None and start < self._data.shape[1]:
                    leftover = self._data[:, start:]
                    data = np.concatenate([leftover, data], axis=1)

                # reset everything
                self._data = data
                self._frame_idx = 0
                start, stop = 0, self.chunk_size

        # create a package using the data, using
        # the internal index to set the request
        # id for downstream processes to reconstruct
        # the order in which data should be processed
        x = self._data[:, start:stop]
        package = Package(
            x=x,
            t0=time.time(),
            sequence_id=self.sequence_id,
            request_id=self._idx,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )

        # increment the request index for the next request
        self._idx += 1
        self._frame_idx += 1

        return package

    def process(self, package):
        super().process(package)

        # if this is the last package in the sequence,
        # raise a StopIteration so that downstream
        # processes know that we're done here
        if package.sequence_end:
            raise StopIteration
