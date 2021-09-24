import sys
from typing import List, Optional, Sequence, Tuple, Union

import pytest

from hermes.typeo import typeo


@typeo
def func(a: int, b: int = 10):
    return a + b


@typeo("named-program")
def named(a: int, b: int):
    return a + b


@typeo
def lists_func(a: List[str]):
    return "".join(a)


@typeo
def seq_func(a: Sequence[str]):
    return "".join(a)


@typeo
def tuple_func(a: Tuple[str, ...]):
    return "".join(a)


with pytest.raises(AssertionError):

    @typeo
    def bad_tuple_func(a: Tuple[str, int]):
        return a[0] * a[1]


@typeo
def maybe_list_func(a: Union[str, List[str]]):
    if isinstance(a, str):
        return a
    else:
        return "".join(a)


@typeo
def optional_func(a: int, b: Optional[str] = None):
    return (b or "test") * a


with pytest.raises(ValueError):

    @typeo
    def bad_optional_func(a: Optional[str]):
        return a or "test"


with pytest.raises(TypeError):

    @typeo
    def bad_union_func(a: Union[str, int]):
        return a * 2


with pytest.raises(AssertionError):

    @typeo
    def bad_union_list_func(a: Union[str, List[int]]):
        if isinstance(a, str):
            return a
        else:
            return sum(a)


def test_typeo():
    assert func(1, 2) == 3
    sys.argv = [None, "--a", "1", "--b", "3"]
    assert func() == 4
    sys.argv = [None, "--a", "2"]
    assert func() == 12

    assert lists_func(["t", "e", "s", "t"]) == "test"
    sys.argv = [None, "--a", "t", "e", "s", "t"]
    assert lists_func() == "test"

    assert seq_func(["t", "e", "s", "t"]) == "test"
    assert seq_func() == "test"

    assert tuple_func(["t", "e", "s", "t"]) == "test"
    assert tuple_func() == "test"

    assert maybe_list_func("testing") == "testing"
    assert maybe_list_func(["t", "e", "s", "t"]) == "test"
    assert maybe_list_func() == "test"

    assert optional_func(1, "not none") == "not none"
    assert optional_func(2) == "testtest"
    sys.argv = [None, "--a", "2"]
    assert optional_func() == "testtest"
    sys.argv = [None, "--a", "1", "--b", "not none"]
    assert optional_func() == "not none"

    # TODOs:
    # - capture stdout and make sure that help looks ok
    # - catch empty calls with insufficient args and
    #     make sure the expected error happens
