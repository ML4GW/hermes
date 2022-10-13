import abc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from google.protobuf import text_format
from tritonclient.grpc.model_config_pb2 import ModelConfig

if TYPE_CHECKING:
    from hermes.types import IO_TYPE


@dataclass
class FileSystem(metaclass=abc.ABCMeta):
    root: Union[Path, str]

    @abc.abstractmethod
    def soft_makedirs(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def join(self, *args) -> str:
        pass

    @abc.abstractmethod
    def delete(self):
        pass

    @abc.abstractmethod
    def remove(self, path: str):
        pass

    @abc.abstractmethod
    def list(self, path: Optional[str] = None) -> List[str]:
        pass

    @abc.abstractmethod
    def glob(self, path: str) -> List[str]:
        pass

    @abc.abstractmethod
    def read(self, path: str) -> "IO_TYPE":
        pass

    def read_config(self, path: str) -> ModelConfig:
        config = ModelConfig()
        config_txt = self.read(path)
        text_format.Merge(config_txt, config)
        return config

    @abc.abstractmethod
    def write(self, obj: "IO_TYPE", path: str) -> None:
        pass

    def write_config(self, config: ModelConfig, path: str) -> None:
        self.write(str(config), path)
