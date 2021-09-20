import typing

IO_TYPE = typing.Union[str, bytes]
SHAPE_TYPE = tuple[typing.Optional[int], ...]
EXPOSED_TYPE = typing.Literal["input", "output"]
