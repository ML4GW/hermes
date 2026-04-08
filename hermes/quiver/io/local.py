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
        if not isinstance(self.root, Path):
            self.root = Path(self.root)
        self.root = self.root.resolve()

        self.soft_makedirs("")

    def soft_makedirs(self, path: str):
        target = self.root / path
        existed = target.exists()
        target.mkdir(parents=True, exist_ok=True)
        return not existed

    def join(self, *args):
        path = Path(args[0])
        for part in args[1:]:
            path = path / part

        return str(path)

    def isdir(self, path: str) -> bool:
        return (self.root / path).isdir()

    def list(self, path: Optional[str] = None) -> List[str]:
        target = self.root if path is None else self.root / path
        return [p.name for p in target.iterdir()]

    def glob(self, path: str):
        matches = self.root.glob(path)
        return [str(p.relative_to(self.root)) for p in matches]

    def remove(self, path: str):
        target = self.root / path
        if target.isdir():
            shutil.rmtree(target)
        elif target.is_file():
            target.unlink()
        else:
            matches = self.glob(path)
            if not matches:
                raise NoFilesFoundError(path)
            for match in matches:
                self.remove(match)

    def delete(self):
        self.remove("")

    def read(self, path: str, mode: str = "r") -> "IO_TYPE":
        with open(self.root / path, mode) as f:
            return f.read()

    def write(self, obj: "IO_TYPE", path: str) -> None:

        if isinstance(obj, str):
            mode = "w"
        elif isinstance(obj, bytes):
            mode = "wb"
        else:
            raise TypeError(
                "Expected object to be of type "
                "str or bytes, found type {}".format(type(obj))
            )

        with open(self.root / path, mode) as f:
            f.write(obj)

    def __str__(self):
        return self.root
