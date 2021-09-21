from typing import Literal, Optional, Tuple, Union

IO_TYPE = Union[str, bytes]
SHAPE_TYPE = Tuple[Optional[int], ...]
EXPOSED_TYPE = Literal["input", "output"]
