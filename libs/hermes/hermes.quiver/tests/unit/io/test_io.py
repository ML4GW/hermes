import os
import shutil

import pytest

from hermes.quiver.io import GCSFileSystem, LocalFileSystem


def run_file_manipulations(fs):
    # define some files to create in the
    # filesystem. Important to first make
    # sure that `fs.join` functions properly
    fnames = [
        fs.join("test", "123", "test.txt"),
        fs.join("test", "456", "test.txt"),
        fs.join("test", "test.txt"),
        fs.join("test", "test.csv"),
    ]
    fs.soft_makedirs(fs.join("test", "123"))
    fs.soft_makedirs(fs.join("test", "456"))

    # check to make sure file writing is successful
    for f in fnames:
        fs.write("testing", f)

    assert fs.isdir("test")
    assert fs.isdir(fs.join("test", "123"))
    assert not fs.isdir(fs.join("test", "123", "test.txt"))

    # check that the files were created
    # do not enforce any ordering expecations
    # since os.listdir doesn't enforce any
    results = fs.list("test")
    for f in ["123", "456", "test.csv", "test.txt"]:
        results.remove(f)
    assert len(results) == 0

    # check our glob functionality by making
    # sure that the csv isn't picked up
    expected_name = fnames[2]
    assert fs.glob(fs.join("test", "*.txt")) == [expected_name]

    # confirm that writing was done properly
    # and that reading is functional
    for f in fnames:
        assert fs.read(f) == "testing"

    # remove the csv and then make sure that
    # trying to read it raises an error
    fs.remove(fnames[-1])
    with pytest.raises(FileNotFoundError):
        fs.read(fnames[-1])


def test_local_filesytem():
    # create a local filesystem and
    # verify that it exists
    dirname = "hermes-quiver-test"
    fs = LocalFileSystem(dirname)
    assert os.path.isdir(dirname)

    # run checks in a try-catch in case
    # anything fails so we can delete the
    # directory if anything goes wrong
    try:
        # make sure that paths are joined correctly
        assert fs.join("test", "123") == os.path.join("test", "123")

        # make sure the file system can manipulate
        # files in the expected way
        run_file_manipulations(fs)

        # delete the file system and verify
        # that it no longer exists
        fs.delete()
        assert not os.path.isdir(dirname)
    except Exception:
        # if anything went wrong, explicitly
        # delete the temporary directory with
        # a tried and true method
        shutil.rmtree(dirname)
        raise


@pytest.mark.gcs
def test_gcs_filesystem():
    from google.api_core.exceptions import NotFound

    bucket_name = "hermes-quiver-test"

    # create the bucket file system and
    # run tests in a try-catch in case
    # anything goes wrong to delete the bucket
    fs = GCSFileSystem(bucket_name)
    try:
        # make sure that soft_makedirs
        # doesn't do anything
        assert not fs.soft_makedirs("")

        # make sure that path joining
        # works as expected
        assert fs.join("testing", "123") == "testing/123"

        # make sure the file system can manipulate
        # files in the expected way
        run_file_manipulations(fs)

        # delete the bucket and verify that
        # it no longer exists
        fs.delete()
        with pytest.raises(NotFound):
            fs.client.get_bucket(bucket_name)
    except Exception:
        # delete the bucket explicitly
        # before exiting
        fs.bucket.delete(force=True)
        raise
