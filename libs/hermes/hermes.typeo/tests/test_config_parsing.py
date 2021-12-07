import argparse
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, Sequence
from unittest.mock import Mock

import pytest
import toml

from hermes.typeo import actions, typeo


def set_argv(*args):
    sys.argv = [None] + list(args)


def simple_func(a: int, b: str):
    return b * a


def simple_func_with_underscore_vars(first_arg: int, second_arg: str):
    return simple_func(first_arg, second_arg)


def simple_func_with_default(a: int, b: str = "foo"):
    return simple_func(a, b)


def simple_boolean_func(a: int, b: bool):
    return a if b else a * 2


def get_boolean_func_with_default(default):
    def simple_boolean_func(a: int, b: bool = default):
        return a if b else a * 2

    return simple_boolean_func


def simple_list_func(a: int, b: Sequence[str]):
    return "-".join([i * a for i in b])


def simple_dict_func(a: int, b: Dict[str, int]):
    return {k: v * a for k, v in b.items()}


@contextmanager
def dump_config(config):
    with open("config.toml", "w") as f:
        toml.dump(config, f)
    yield
    os.remove("config.toml")


@pytest.fixture(params=["bar", None])
def simple_config(request):
    a = 3
    config = {"a": a}
    if request.param is not None:
        config["b"] = request.param

    with dump_config({"typeo": config}):
        yield a, request.param


@pytest.fixture
def simple_config_with_underscores():
    a, b = 3, "bar"
    config = {"typeo": {"first_arg": a, "second_arg": b}}
    with dump_config(config):
        yield a, b


@pytest.fixture(params=[True, False])
def bool_config(request):
    a = 3
    with dump_config({"typeo": {"a": a, "b": request.param}}):
        yield a, request.param


@pytest.fixture
def list_config():
    a = 3
    b = ["thom", "jonny", "phil"]
    with dump_config({"typeo": {"a": a, "b": b}}):
        yield a, b


@pytest.fixture
def dict_config():
    a = 3
    b = {"thom": 1, "jonny": 10, "phil": 99}
    with dump_config({"typeo": {"a": a, "b": b}}):
        yield a, b


@pytest.fixture(params=[1, 2])
def subcommands_config(request):
    config = {
        "typeo": {
            "a": request.param,
            "commands": {
                "command1": OrderedDict([("name", "thom"), ("age", 5)]),
                "command2": OrderedDict(
                    [
                        ("first_name", "jonny"),
                        ("last_name", "greenwood"),
                        ("age", 10),
                    ]
                ),
            },
        }
    }
    with dump_config(config):
        d = config["typeo"]["commands"]["command" + str(request.param)]
        yield request.param, d


def _test_action(expected, bools=None, cmd=None):
    mock = Mock()
    parser = argparse.ArgumentParser(prog="dummy")

    action = parser.add_argument(
        "--foo", action=actions.TypeoTomlAction, bools=bools, nargs="?"
    )
    mock = Mock()

    if cmd is None:
        value = "config.toml"
    else:
        value = "config.toml::" + cmd
    action(None, mock, value)
    assert mock.foo == expected


def test_config(simple_config):
    a, b = simple_config

    parser = argparse.ArgumentParser(prog="dummy")
    with pytest.raises(KeyError):
        parser.add_argument("--foo", action=actions.TypeoTomlAction)

    expected = ["--a", str(a)]
    if b is not None:
        expected += ["--b", b]
    _test_action(expected)

    # now test the behavior of a typeo-ified function
    set_argv("--typeo", "config.toml")
    if b is not None:
        # make sure that the config value of b
        # is correctly set regardless of whether
        # the function uses a default or not
        expected = simple_func(a, b)
        result = typeo(simple_func)()
        assert result == expected

        expected = simple_func_with_default(a, b)
        result = typeo(simple_func_with_default)()
        assert result == expected
    else:
        # we didn't specify b, and simple_func requires it,
        # so this should raise a parsing error -> sys exit
        with pytest.raises(SystemExit):
            typeo(simple_func)()

        # now make sure that the function with a default
        # uses the deafult value when we don't specify b
        expected = simple_func_with_default(a)
        result = typeo(simple_func_with_default)()
        assert expected == result

    # make sure passing extra args when we specify
    # a config raises a ValueError
    with pytest.raises(ValueError):
        set_argv("--typeo", "config.toml", "--a", "10")
        typeo(simple_func)()


@pytest.mark.depends(on=["test_config"])
def test_underscore_variables(simple_config_with_underscores):
    a, b = simple_config_with_underscores

    expected = ["--first-arg", str(a), "--second-arg", str(b)]
    _test_action(expected)

    set_argv("--typeo", "config.toml")
    expected = simple_func_with_underscore_vars(a, b)
    result = typeo(simple_func_with_underscore_vars)()
    assert expected == result


@pytest.mark.depends(on=["test_config"])
def test_config_booleans(bool_config):
    a, config_bool = bool_config

    # if the flag in the config matches the
    # default value passed to TypeoTomlAction,
    # then make sure the flag gets added
    for boolean in [True, False]:
        expected = ["--a", str(a)]
        if boolean != config_bool:
            expected += ["--b"]
        _test_action(expected, bools={"b": boolean})

    # now make sure that a typeo-ified version
    # of the function without a default parses
    # the correct value for the boolean
    set_argv("--typeo", "config.toml")
    result = typeo(simple_boolean_func)()
    expected = simple_boolean_func(a, config_bool)
    assert result == expected

    # now make sure this still happens for both defaults
    for default in [True, False]:
        func = get_boolean_func_with_default(default)
        result = typeo(func)()
        expected = func(a, config_bool)
        assert result == expected


@pytest.mark.depends(on=["test_config"])
def test_config_lists(list_config):
    a, b = list_config
    expected = ["--a", str(a), "--b"]
    expected.extend(b)
    _test_action(expected)

    set_argv("--typeo", "config.toml")
    assert typeo(simple_list_func)() == simple_list_func(a, b)


@pytest.mark.depends(on=["test_config"])
def test_config_dicts(dict_config):
    a, b = dict_config
    expected = ["--a", str(a), "--b"]
    for k, v in b.items():
        expected.append(f"{k}={v}")
    _test_action(expected)

    set_argv("--typeo", "config.toml")
    assert typeo(simple_dict_func)() == simple_dict_func(a, b)


@pytest.mark.depends(on=["test_config"])
def test_subcommands(subcommands_config):
    mock = Mock()

    def base_func(a: int):
        mock.a = a

    def command1(name: str, age: int):
        mock.name = name
        return age * mock.a

    def command2(first_name: str, last_name: str, age: int):
        mock.name = first_name + " " + last_name
        return age * mock.a * 2

    a, d = subcommands_config
    expected = ["--a", str(a), "command" + str(a)]
    for k, v in d.items():
        k = k.replace("_", "-")
        expected.append(f"--{k}")
        expected.append(str(v))
    _test_action(expected, cmd="command" + str(a))

    set_argv("--typeo", "config.toml::command" + str(a))
    result = typeo(base_func, command1=command1, command2=command2)()
    name = " ".join([v for k, v in d.items() if "name" in k])
    assert mock.a == a
    assert mock.name == name
    assert result == d["age"] * mock.a * a
