import sys
from enum import Enum
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import pytest

from hermes.typeo import spoof, typeo


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


def set_argv(*args):
    sys.argv = [None] + list(args)


def test_typeo():
    assert func(1, 2) == 3
    set_argv("--a", "1", "--b", "3")
    assert func() == 4
    set_argv("--a", "2")
    assert func() == 12

    assert lists_func(["t", "e", "s", "t"]) == "test"
    set_argv("--a", "t", "e", "s", "t")
    assert lists_func() == "test"

    assert seq_func(["t", "e", "s", "t"]) == "test"
    assert seq_func() == "test"

    assert tuple_func(["t", "e", "s", "t"]) == "test"
    assert tuple_func() == "test"

    assert optional_func(1, "not none") == "not none"
    assert optional_func(2) == "testtest"
    set_argv("--a", "2")
    assert optional_func() == "testtest"
    set_argv("--a", "1", "--b", "not none")
    assert optional_func() == "not none"

    # TODOs:
    # - capture stdout and make sure that help looks ok
    # - catch empty calls with insufficient args and
    #     make sure the expected error happens


@pytest.mark.depends(on=["test_typeo"])
@pytest.mark.parametrize("annotation", [List, Iterable, Tuple, Sequence])
def test_maybe_sequence_funcs(annotation):
    @typeo
    def maybe_func(a: Union[str, annotation[str]]):
        if isinstance(a, str):
            return a + " no sequence"
        else:
            return "".join(a) + " yes sequence"

    assert maybe_func("testing") == "testing no sequence"
    assert maybe_func(["t", "e", "s", "t"]) == "test yes sequence"

    set_argv("--a", *"test")
    assert maybe_func() == "test yes sequence"

    with pytest.raises(TypeError):

        @typeo
        def bad_maybe_func(a: Union[str, annotation[int]]):
            return


@pytest.mark.depends(on=["test_typeo"])
def test_subparsers():
    d = {}

    def f1(a: int, b: int):
        return a + b

    def f2(a: int, c: int):
        return a - c

    @typeo(add=f1, subtract=f2)
    def f(i: int):
        d["f"] = i

    set_argv("--i", "2", "add", "--a", "1", "--b", "2")
    assert f() == 3
    assert d["f"] == 2

    set_argv("--i", "4", "subtract", "--a", "9", "--c", "3")
    assert f() == 6
    assert d["f"] == 4


@pytest.mark.depends(on=["test_typeo"])
def test_enums():
    class Member(Enum):
        SINGER = "Thom"
        GUITAR = "Jonny"
        DRUMS = "Phil"

    # make sure that the parsed value comes
    # out as the appropriate Enum instance
    @typeo
    def f(member: Member):
        return member

    set_argv("--member", "Thom")
    assert f() == Member.SINGER

    # make sure that it's argparse that
    # catches if the choice is invalid
    set_argv("--member", "error")
    with pytest.raises(SystemExit):
        f()

    # make sure that sequences of enums get
    # mapped to lists of the Enum instances
    @typeo
    def f(members: Sequence[Member]):
        return members

    set_argv("--members", "Thom", "Thom", "Jonny")
    assert f() == [Member.SINGER, Member.SINGER, Member.GUITAR]


@pytest.mark.depends(on=["test_typeo"])
@pytest.mark.parametrize("generic", [List, Tuple, Sequence])
def test_blank_generics(generic):
    """Untyped generics should default to parsing as strings"""

    @typeo
    def blank_generic_func(a: generic):
        return [i + "a" for i in a]

    args = ["test", "one", "two"]
    set_argv("--a", *args)

    assert blank_generic_func() == [i + "a" for i in args]

    set_argv("--a", *"123")
    assert blank_generic_func() == ["1a", "2a", "3a"]


@pytest.mark.depends(on=["test_maybe_sequence_funcs", "test_blank_generics"])
@pytest.mark.parametrize("generic", [List, Tuple, Sequence])
def test_unions_with_blank_generics(generic):
    """Test generics used as the second argument to a Union

    Generic sequence types should default to the type
    of the first argument of the Union when used with
    a Union.
    """

    @typeo
    def blank_generic_func(a: Union[str, generic]):
        return [i + "a" for i in a]

    args = ["test", "one", "two"]
    set_argv("--a", *args)

    result = blank_generic_func()

    # TODO: uncomment this when we implement
    # action for mapping to tuples
    # assert isinstance(result, generic)
    assert isinstance(result, Sequence)
    assert result == [i + "a" for i in args]

    @typeo
    def blank_generic_func(a: Union[int, generic]):
        return [i + 2 for i in a]

    set_argv("--a", *"123")
    assert blank_generic_func() == [3, 4, 5]


@pytest.mark.depends(on=["test_typeo"])
def test_callables():
    import math

    @typeo
    def func_of_func(f: Callable):
        return f(3, 2)

    assert func_of_func(divmod) == (1, 1)
    set_argv("--f", "math.pow")
    assert func_of_func() == 9

    @typeo
    def func_of_funcs(fs: Sequence[Callable]):
        return sum([f(3) for f in fs])

    answer = math.sqrt(3) + math.log(3)
    assert func_of_funcs([math.sqrt, math.log]) == answer

    set_argv("--fs", "math.sqrt", "math.log")
    assert func_of_funcs() == answer

    with pytest.raises(SystemExit):
        set_argv("--f", "bad.libary.name")
        func_of_func()


@pytest.mark.depends(on=["test_typeo"])
def test_spoof():
    def simple_func(a: int, b: str):
        return b * a

    result = spoof(simple_func, "--a", "2", "--b", "cat")
    assert result["a"] == 2
    assert result["b"] == "cat"

    with pytest.raises(SystemExit):
        spoof(simple_func, "--a", "2")

    with pytest.raises(ValueError):
        spoof(simple_func, "--a", "2", filename="pyproject.toml")
