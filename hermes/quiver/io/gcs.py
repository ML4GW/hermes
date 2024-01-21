from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

try:
    from google.api_core.exceptions import Forbidden, NotFound
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import storage

    _has_google_libs = True
except ImportError:
    _has_google_libs = False

from hermes.quiver.io.exceptions import NoFilesFoundError
from hermes.quiver.io.file_system import FileSystem

if TYPE_CHECKING:
    from hermes.quiver.types import IO_TYPE


@dataclass
class GCSFileSystem(FileSystem):
    credentials: Optional[str] = None

    def __post_init__(self):
        if not _has_google_libs:
            raise ImportError(
                "Must install google-cloud-storage to use GCSFileSystem"
            )

        if self.credentials is not None:
            self.client = storage.Client.from_service_account_json(
                self.credentials
            )
        else:
            try:
                self.client = storage.Client()
            except DefaultCredentialsError:
                raise ValueError(
                    "Must specify service account json file "
                    "via the `GOOGLE_APPLICATION_CREDENTIALS` "
                    "environment variable to use a GCSFileSystem"
                )

        # split the bucket name from the rest
        # of the path, if one was specified
        # TODO: need to verify that you can point
        # triton to paths deeper than the root
        # level of a bucket
        split_path = self.root.split("/", maxsplit=1)
        try:
            bucket_name, prefix = split_path
            self.root = prefix.rstrip("/")
        except ValueError:
            bucket_name = split_path[0]
            self.root = ""

        try:
            # try to get the bucket if it already exists
            self.bucket = self.client.get_bucket(bucket_name)
        except NotFound:
            # if it doesn't exist, try to create it
            # note that we don't need to worry about
            # name collisions because this would have
            # raised `Forbidden` rather than `NotFound`
            self.bucket = self.client.create_bucket(bucket_name)
        except Forbidden:
            # bucket already exists but the given
            # credentials are insufficient to access it
            raise ValueError(
                "Provided credentials are unable to access "
                f"GCS bucket with name {bucket_name}"
            )

    def soft_makedirs(self, path: str):
        """
        Does nothing in the context of a GCS bucket
        where objects need to be created one
        by one
        """
        return False

    def join(self, *args) -> str:
        for arg in args:
            if not isinstance(arg, str):
                raise TypeError(
                    "join() argument must be str, not {}".format(type(arg))
                )
        return "/".join(args)

    def isdir(self, path: str) -> bool:
        if self.root:
            path = self.join(self.root, path)

        for blob in self.bucket.list_blobs(prefix=path):
            # only wait until we find the first
            # instance of this path being used
            # as a directory to return True
            if "/" in blob.name.replace(path, "", 1):
                return True

    def list(self, path: Optional[str] = None) -> List[str]:
        if path is not None and self.root:
            # we specified a path, and we have a root,
            # so join them to make the prefix
            path = self.join(self.root, path)
        elif path is None:
            # we didn't specify a path, so we're
            # just listing the root level
            path = self.root

        # rstrip off the path separator for consistency,
        # in case the user passed a path ending in "/"
        path = path.rstrip("/")

        # directories we need to keep track of as
        # a set since they can obviously have multiple
        # objects inside of them
        fs, dirs = [], set()
        for blob in self.bucket.list_blobs(prefix=path):
            name = blob.name.replace(path, "", 1).strip("/")
            try:
                # check if this is a directory by seeing
                # if there is at least one path separator
                # in the remaining name of the blob
                d, _ = name.split("/", maxsplit=1)
                dirs.add(d)
            except ValueError:
                # otherwise assume this is a file
                fs.append(name)

        # sort everything and return
        return sorted(list(dirs)) + sorted(fs)

    def glob(self, path: str):
        postfix = None
        prefix = self.root
        if "*" in path and path != "*":
            splits = path.split("*")
            if len(splits) > 2:
                raise ValueError(f"Could not parse path {path}")

            _prefix, postfix = splits
            if _prefix and prefix:
                prefix = self.join(prefix, _prefix)
            elif _prefix:
                prefix = _prefix
        else:
            prefix = path

        names = []
        for blob in self.bucket.list_blobs(prefix=prefix):
            if postfix is not None:
                name = blob.name.replace(prefix, "")
                name = name.strip("/")

                if name.endswith(postfix) and "/" not in name:
                    names.append(blob.name)
            else:
                names.append(blob.name)
        return names

    def remove(self, path: str):
        names = self.glob(path)
        if len(names) == 0:
            raise NoFilesFoundError(path)

        for name in names:
            blob = self.bucket.get_blob(name)
            blob.delete()

    def delete(self):
        self.bucket.delete(force=True)

    def read(self, path: str, mode: str = "r") -> "IO_TYPE":
        if self.root:
            path = self.join(self.root, path)

        blob = self.bucket.get_blob(path)
        if blob is None:
            raise FileNotFoundError(path)

        content = blob.download_as_bytes()
        if mode == "r":
            content = content.decode()
        return content

    def write(self, obj: "IO_TYPE", path: str) -> None:
        if self.root:
            path = self.join(self.root, path)

        blob = self.bucket.get_blob(path)
        if blob is not None:
            blob.delete()
        blob = self.bucket.blob(path)

        if isinstance(obj, str):
            content_type = "text/plain"
        elif isinstance(obj, bytes):
            content_type = "application/octet-stream"
        else:
            raise TypeError(
                "Expected object to be of type "
                "str or bytes, found type {}".format(type(obj))
            )

        blob.upload_from_string(obj, content_type=content_type)

    def __str__(self):
        return f"gs://{self.bucket}/{self.root}"
