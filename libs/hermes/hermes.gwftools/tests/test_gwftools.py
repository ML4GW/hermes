import os
import time
from math import ceil, floor

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


def test_frame_crawler(tstamp, fformat):
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
            super().process(fname)

    writer = FrameWriter(name="writer", rate=10)
    crawler = gwf.FrameCrawler("tmp", N=10, name="crawler")

    try:
        with writer:
            time.sleep(0.15)
            with crawler:
                i = 0
                for f1, f2 in zip(writer, crawler):
                    assert f1 == f2
                    os.remove(f1)
                    i += 1
                assert i == 10
    finally:
        write_dir = "tmp"
        if os.path.exists(write_dir):
            for f in os.listdir(write_dir):
                os.remove(os.path.join(write_dir, f))
            os.rmdir(write_dir)
