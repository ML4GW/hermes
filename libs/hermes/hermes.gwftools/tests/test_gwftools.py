import os
import time
from math import ceil, floor

import numpy as np
import pytest

from hermes.gwftools import gwftools as gwf
from hermes.quiver.io import GCSFileSystem
from hermes.stillwater import PipelineProcess


def test_parse_frame_name():
    fname = "H-H1_llhoft-1313883327-1.gwf"
    tstamp, length = gwf._parse_frame_name(fname)
    assert tstamp == 1313883327
    assert length == 1

    fname = "H-H1_llhoft-1313883327-4096.gwf"
    tstamp, length = gwf._parse_frame_name(fname)
    assert tstamp == 1313883327
    assert length == 4096


@pytest.fixture(scope="session")
def tstamp():
    return 1313883327


@pytest.fixture(scope="session")
def fformat():
    return "H-H1_llhoft-{}-1.gwf"


@pytest.fixture(scope="session", params=[None, "frames"])
def bucket_name(request, tstamp, fformat):
    bucket_name = "gwftools-test-bucket"
    if request.param is not None:
        bucket_name += "/" + request.param
    fs = GCSFileSystem(bucket_name)

    for i in range(10):
        fs.write("", fformat.format(tstamp + i))

    yield bucket_name
    fs.delete()


@pytest.mark.gcs
@pytest.mark.parametrize(
    "write_dir,t0,length",
    [
        (None, None, None),
        ("tmp", None, None),
        (None, 3, None),
        (None, 3, 4),
        (None, 3.5, None),
        (None, 3.5, 4.5),
        pytest.param(
            None, None, 4, marks=pytest.mark.xfail(raises=ValueError)
        ),  # fails because specify length but not t0
        pytest.param(
            None,
            11,
            None,
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),  # fails because no frames before t0
        pytest.param(
            None,
            11,
            4,
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),  # fails because no frames in range
    ],
)
def test_gcs_downloader(write_dir, t0, length, bucket_name, tstamp, fformat):
    try:
        with gwf.GCSFrameDownloader(
            root=bucket_name,
            t0=tstamp + t0 if t0 is not None else t0,
            length=length,
            write_dir=write_dir,
            name="downloader",
        ) as downloader:
            if write_dir is not None:
                assert os.path.exists(write_dir)

            i = floor(t0) if t0 is not None else 0
            for fname in downloader:
                assert os.path.exists(fname)
                assert os.path.basename(fname) == fformat.format(tstamp + i)

                i += 1
                os.remove(fname)

            if length is not None:
                frames_seen = i - floor(t0)
                expected = ceil(length)
                assert frames_seen == expected
    finally:
        if write_dir is not None and os.path.exists(write_dir):
            for f in os.listdir(write_dir):
                os.remove(os.path.join(write_dir, f))
            os.rmdir(write_dir)


@pytest.mark.parametrize(
    "timeout",
    [
        None,
        pytest.param(
            0.05, marks=pytest.mark.xfail(raises=RuntimeError, strict=True)
        ),
    ],
)
def test_frame_crawler(timeout, tstamp, fformat):
    # start by building a dummy process which writes
    # "frame" files in chronological order
    class FrameWriter(PipelineProcess):
        def __init__(self, *args, **kwargs):
            self.i = 0
            self.tstamp = tstamp
            self.fformat = fformat
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            super().__init__(*args, **kwargs)

        def get_package(self):
            fname = self.fformat.format(self.tstamp + self.i)
            self.i += 1
            return os.path.join("tmp", fname)

        def process(self, fname):
            with open(fname, "w") as f:
                f.write("")

            # send the filename to the out q
            # so that we can match it up with
            # what the crawler outputs
            super().process(fname)

    writer = FrameWriter(name="writer", rate=10)
    crawler = gwf.FrameCrawler("tmp", N=10, timeout=timeout, name="crawler")

    try:
        # start the writer first so that the crawler
        # has files to find to learn its pattern
        with writer:
            # sleep to make sure at least one frame
            # gets written first
            time.sleep(0.15)
            with crawler:
                # now iterate through their respective
                # outputs and make sure that they match
                # keep track of how many we've done to
                # verify that the StopIteration gets
                # raised in the right place
                i = 0
                for f1, f2 in zip(writer, crawler):
                    assert f1 == f2

                    # delete files once we're done with them
                    os.remove(f1)
                    i += 1
                assert i == 10
    finally:
        # delete the temporary local write dir
        # no matter what happens. Make sure to
        # clear all the files first
        write_dir = "tmp"
        if os.path.exists(write_dir):
            for f in os.listdir(write_dir):
                os.remove(os.path.join(write_dir, f))
            os.rmdir(write_dir)


@pytest.fixture
def fnames(tstamp, fformat):
    from gwpy.timeseries import TimeSeries

    write_dir = "tmp"
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    x = np.zeros((4096,))
    fnames = []
    for i in range(10):
        ts = TimeSeries(x + i, t0=tstamp + i, sample_rate=4096, channel="x")

        fname = os.path.join(write_dir, fformat.format(tstamp + i))
        ts.write(fname)
        fnames.append(fname)

    yield fnames
    if os.path.exists(write_dir):
        for f in os.listdir(write_dir):
            os.remove(os.path.join(write_dir, f))
        os.rmdir(write_dir)


@pytest.mark.gwf
@pytest.mark.parametrize(
    "chunk_size,step_size,sample_rate",
    [
        (256, 256, 4096),
        (256, 256, 2048),
        (256, 32, 4096),
        (256, 31, 4096),
    ],
)
def test_frame_loader(chunk_size, step_size, sample_rate, fnames):
    with gwf.FrameLoader(
        chunk_size=chunk_size,
        step_size=step_size,
        sample_rate=sample_rate,
        channels=["x"],
        remove=True,
        name="loader",
    ) as loader:
        for fname in fnames:
            loader.in_q.put(fname)
        loader.in_q.put(StopIteration)

        expected_array = np.repeat(np.arange(10), 4096)
        for i, package in enumerate(loader):
            assert package.request_id == i
            assert package.sequence_start == (i == 0)

            x = package.x
            assert x.shape == (1, chunk_size)

            expected_value = expected_array[
                i * step_size : i * step_size + chunk_size
            ]
            assert (x[0] == expected_value).all()

        assert package.sequence_end
        assert i == (sample_rate * 10 // step_size)

    assert len(os.listdir("tmp")) == 0
