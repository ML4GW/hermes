import glob
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from hermes.quiver.io.exceptions import NoFilesFoundError
from hermes.quiver.io.file_system import FileSystem

if TYPE_CHECKING:
    from hermes.types import IO_TYPE


@dataclass
class LocalFileSystem(FileSystem):
    def __post_init__(self):
        # TODO: switch to preferring Path as root
        # and using pathlib apis instead of os.
        if not isinstance(self.root, Path):
            self.root = Path(self.root)
        self.root = str(self.root.resolve())

        self.soft_makedirs("")

    def soft_makedirs(self, path: str):
        path = self.join(self.root, path)

        # TODO: start using exists_ok kwargs once
        # we know we have the appropriate version
        if not os.path.exists(path):
            os.makedirs(path)
            return True
        return False

    def join(self, *args):
        return os.path.join(*args)

    def isdir(self, path: str) -> bool:
        path = self.join(self.root, path)
        return os.path.isdir(path)

    def list(self, path: Optional[str] = None) -> List[str]:
        if path is not None:
            path = self.join(self.root, path)
        else:
            path = self.root
        return os.listdir(path)

    def glob(self, path: str):
        files = glob.glob(self.join(self.root, path))

        # get rid of the root to put everything
        # relative to the fs root
        if self.root.endswith(os.path.sep):
            prefix = self.root
        else:
            prefix = self.root + os.path.sep

        return [f.replace(prefix, "") for f in files]

    def remove(self, path: str):
        path = self.join(self.root, path)

        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
        else:
            paths = self.glob(path)
            if len(paths) == 0:
                raise NoFilesFoundError(path)

            for path in paths:
                self.remove(path)

    def delete(self):
        self.remove("")

    def read(self, path: str, mode: str = "r") -> "IO_TYPE":
        path = self.join(self.root, path)
        with open(path, mode) as f:
            return f.read()

    def write(self, obj: "IO_TYPE", path: str) -> None:
        path = self.join(self.root, path)

        if isinstance(obj, str):
            mode = "w"
        elif isinstance(obj, bytes):
            mode = "wb"
        else:
            raise TypeError(
                "Expected object to be of type "
                "str or bytes, found type {}".format(type(obj))
            )

        with open(path, mode) as f:
            f.write(obj)

    def __str__(self):
        return self.root
