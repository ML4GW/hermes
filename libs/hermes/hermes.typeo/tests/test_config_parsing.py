import argparse
import os
import shutil
import sys
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, Sequence
from unittest.mock import Mock

import pytest
import toml

from hermes.typeo import actions, spoof, typeo


@pytest.fixture(scope="module", params=[None, ".", "config.toml"])
def fname(request):
    return request.param


@pytest.fixture(scope="module")
def set_argv(fname):
    def fn(*args):
        if fname is None:
            sys.argv = [None, "--typeo"]
        else:
            sys.argv = [None, "--typeo", fname]

        if len(args) == 0:
            return

        args = list(args)
        if args[0].startswith(":") and len(sys.argv) > 2:
            sys.argv[-1] += args.pop(0)
        sys.argv.extend(args)

    return fn


@pytest.fixture(
    scope="module",
    params=[False, pytest.param(True, marks=pytest.mark.xfail)],
)
def format_wrong(request):
    return request.param


def simple_func(a: int, b: str):
    return b * a


def simple_func_with_default(a: int, b: str = "foo"):
    return simple_func(a, b)


def simple_func_with_underscore_vars(first_arg: int, second_arg: str):
    return simple_func(first_arg, second_arg)


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
def dump_config(config, fname, format_wrong):
    if fname is None or os.path.isdir(fname):
        # if fname is None, want to put pyproject.toml in the cwd
        dirname = fname or "."
        fname = os.path.join(dirname, "pyproject.toml")

        # make sure we store the existing pyproject.toml
        # somewhere safe before we write to it
        if os.path.exists(fname):
            move_back = True
            dummy_fname = "_" + os.path.basename(fname)
            shutil.move(fname, os.path.join(dirname, dummy_fname))
        else:
            move_back = False

        if not format_wrong:
            # since this will point to a pyproject.toml,
            # the assumption is that the config will follow
            # the guideline that tool-specific configs
            # will fall under the "tool" section
            config = {"tool": config}
    elif format_wrong:
        # vice versa if we're not talking about a pyproject.toml
        config = {"tool": config}

    # write the config to the specified file
    with open(fname, "w") as f:
        toml.dump(config, f)

    try:
        yield
    finally:
        # not sure if we need the try-catch, but being extra safe
        if os.path.basename(fname) != "pyproject.toml":
            # for non-pyprojects, just delete the temporary file
            os.remove(fname)
        elif move_back:
            # for pyprojects, make sure to move the original
            # pyproject.toml back to its location
            shutil.move(os.path.join(dirname, dummy_fname), fname)


@pytest.fixture
def a():
    return 3


@pytest.fixture(params=[True, False])
def use_env_var(request):
    return request.param


@pytest.fixture(params=["bar", None])
def simple_config(request, use_env_var, a):
    if use_env_var:
        os.environ["a"] = str(a)
        config = {"a": "${a}"}
    else:
        config = {"a": a}

    if request.param is not None:
        config["b"] = request.param
    return config


@pytest.fixture
def simple_config_no_fail(simple_config, fname, a):
    """Version of simple_config that won't fail

    Need this so that we can have tests depend on
    test_config without skipping due to xfails.
    """

    with dump_config({"typeo": simple_config}, fname, False):
        yield a, simple_config.get("b", None)


@pytest.fixture
def simple_config_with_fail(simple_config, fname, format_wrong, a):
    """Version of simple config that can fail

    Still want to verify that improperly formatted configs
    fail at a simple level, so create a version for a test
    that can fail that we won't depend on
    """

    with dump_config({"typeo": simple_config}, fname, format_wrong):
        yield a, simple_config.get("b", None)


@pytest.fixture
def simple_config_with_section(simple_config, fname, format_wrong, a):
    """Move the `a` argument of simple config to a `script section"""

    simple_config["scripts"] = {"foo": {"a": simple_config.pop("a")}}
    with dump_config({"typeo": simple_config}, fname, format_wrong):
        yield a, simple_config.get("b", None)


@pytest.fixture
def simple_config_with_underscores(fname, format_wrong, a, use_env_var):
    b = "bar"

    config = {"first_arg": a}
    if use_env_var:
        os.environ["SECOND_ARG"] = b
        config["second_arg"] = "${SECOND_ARG}"
    else:
        config["second_arg"] = b

    with dump_config({"typeo": config}, fname, format_wrong):
        yield a, b


@pytest.fixture(params=[True, False])
def bool_config(request, fname, format_wrong):
    a = 3
    with dump_config(
        {"typeo": {"a": a, "b": request.param}}, fname, format_wrong
    ):
        yield a, request.param


@pytest.fixture
def list_config(fname, format_wrong, use_env_var, a):
    config = {"a": a}
    b = ["thom", "jonny", "phil"]
    if use_env_var:
        os.environ["VOCALS"] = "thom"
        os.environ["DRUMS"] = "phil"
        config["b"] = ["${VOCALS}", "jonny", "${DRUMS}"]
    else:
        config["b"] = b

    with dump_config({"typeo": config}, fname, format_wrong):
        yield a, b


@pytest.fixture
def dict_config(fname, format_wrong, use_env_var, a):
    b = {"thom": 1, "jonny": 10, "phil": 99}
    config = {"a": a}
    if use_env_var:
        os.environ["THOM"] = "1"
        os.environ["phil"] = "99"
        config["b"] = {"thom": "${THOM}", "jonny": 10, "phil": "${phil}"}
    else:
        config["b"] = b

    with dump_config({"typeo": {"a": a, "b": b}}, fname, format_wrong):
        yield a, b


@pytest.fixture(params=[1, 2])
def command(request):
    return f"command{request.param}"


@pytest.fixture
def subcommands_config(command):
    return {
        "typeo": {
            "a": int(command[-1]),
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


@pytest.fixture
def subcommands_config_no_sections(
    subcommands_config, command, fname, format_wrong
):
    with dump_config(subcommands_config, fname, format_wrong):
        d = subcommands_config["typeo"]["commands"][command]
        yield command, d


@pytest.fixture
def subcommands_with_sections_config(
    subcommands_config, command, fname, format_wrong
):
    commands = subcommands_config["typeo"].pop("commands")
    subcommands_config["typeo"]["scripts"] = {"foo": {"commands": commands}}
    with dump_config(subcommands_config, fname, format_wrong):
        d = subcommands_config["typeo"]["scripts"]["foo"]["commands"][command]
        yield command, d


def _test_action(expected, fname, bools=None, section=None, cmd=None):
    """Test the TypeoTomlAction directly

    Make sure that the TypeoTomlAction, instantiated with
    the specified `bools`, correctly parses
    the config at `fname` by matching its result to
    the one specified by `expected`, including any args
    passed in a specific `section` or by a `cmd`.
    """

    mock = Mock()
    parser = argparse.ArgumentParser(prog="dummy")

    action = parser.add_argument(
        "--foo", action=actions.TypeoTomlAction, bools=bools, nargs="?"
    )
    mock = Mock()

    value = fname
    if section is not None:
        value = value or ""
        value += ":" + section
        if cmd is not None:
            value += ":" + cmd
    elif cmd is not None:
        value = value or ""
        value += "::" + cmd

    # make sure the parsed value assigned to the mock's
    # `foo` attribute matches up with the expected value
    action(None, mock, value)
    assert mock.foo == expected


def test_simple_failure(simple_config_with_fail, fname):
    a, b = simple_config_with_fail
    expected = ["--a", str(a)]
    if b is not None:
        expected += ["--b", b]
    _test_action(expected, fname)


def test_config(simple_config_no_fail, fname, set_argv):
    a, b = simple_config_no_fail

    parser = argparse.ArgumentParser(prog="dummy")
    with pytest.raises(KeyError):
        parser.add_argument("--foo", action=actions.TypeoTomlAction)

    expected = ["--a", str(a)]
    if b is not None:
        expected += ["--b", b]
    _test_action(expected, fname)

    # now test the behavior of a typeo-ified function
    set_argv()
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
        set_argv("--a", "10")
        typeo(simple_func)()

    # make sure that subcommands with no arguments
    # run successfully even if the command section
    # is absent
    def subcommand():
        return "foo"

    set_argv("::sub")
    func = simple_func if b is not None else simple_func_with_default
    assert typeo(func, sub=subcommand)() == "foo"

    # subcommands with only default values should be
    # able to execute, even if the commands section
    # is missing from the config
    def subcommand_with_defaults(c: int = 3):
        return c + 1

    assert typeo(func, sub=subcommand_with_defaults)() == 4

    # finally, subcommands with required arguments should fail
    # if the commands section of the config is missing
    def bad_subcommand(c: int):
        return c + 1

    with pytest.raises(SystemExit):
        typeo(func, sub=bad_subcommand)()


@pytest.mark.depends(on=["test_config"])
def test_underscore_variables(simple_config_with_underscores, fname, set_argv):
    a, b = simple_config_with_underscores

    expected = ["--first-arg", str(a), "--second-arg", str(b)]
    _test_action(expected, fname)

    set_argv()
    expected = simple_func_with_underscore_vars(a, b)
    result = typeo(simple_func_with_underscore_vars)()
    assert expected == result


@pytest.mark.depends(on=["test_config"])
def test_config_booleans(bool_config, fname, set_argv):
    a, config_bool = bool_config

    # if the flag in the config matches the
    # default value passed to TypeoTomlAction,
    # then make sure the flag gets added
    for boolean in [True, False]:
        expected = ["--a", str(a)]
        if boolean != config_bool:
            expected += ["--b"]
        _test_action(expected, fname, bools={"b": boolean})

    # now make sure that a typeo-ified version
    # of the function without a default parses
    # the correct value for the boolean
    set_argv()
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
def test_config_lists(list_config, fname, set_argv):
    a, b = list_config
    expected = ["--a", str(a), "--b"]
    expected.extend(b)
    _test_action(expected, fname)

    set_argv()
    assert typeo(simple_list_func)() == simple_list_func(a, b)


@pytest.mark.depends(on=["test_config"])
def test_config_dicts(dict_config, fname, set_argv):
    a, b = dict_config
    expected = ["--a", str(a), "--b"]
    for k, v in b.items():
        expected.append(f"{k}={v}")
    _test_action(expected, fname)

    set_argv()
    assert typeo(simple_dict_func)() == simple_dict_func(a, b)


@pytest.mark.depends(on=["test_config"])
def test_script_sections(simple_config_with_section, fname, set_argv):
    a, b = simple_config_with_section

    expected = ["--b", b] if b is not None else []
    expected += ["--a", str(a)]
    _test_action(expected, fname, section="foo")

    set_argv(":foo")
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

    set_argv(":bar")
    with pytest.raises(SystemExit):
        typeo(simple_func)()

    def subcommand():
        return 4

    set_argv(":foo:sub")
    func = simple_func if b is not None else simple_func_with_default
    assert typeo(func, sub=subcommand)() == 4

    def subcommand_with_default(c: int = 4):
        return c + 1

    assert typeo(func, sub=subcommand_with_default)() == 5

    def bad_subcommand(c: int):
        return c + 1

    with pytest.raises(SystemExit):
        typeo(func, sub=bad_subcommand)()


class SubcommandsTester:
    mock = Mock()

    def base_func(self, a: int):
        self.mock.a = a

    def command1(self, name: str, age: int):
        self.mock.name = name
        return age * self.mock.a

    def command2(self, first_name: str, last_name: str, age: int):
        self.mock.name = first_name + " " + last_name
        return age * self.mock.a * 2

    def test_with_config(
        self, command, command_dict, fname, set_argv, section=None
    ):
        a = int(command[-1])
        expected = ["--a", str(a), command]
        for k, v in command_dict.items():
            k = k.replace("_", "-")
            expected.append(f"--{k}")
            expected.append(str(v))
        _test_action(expected, fname, cmd=command, section=section)

        section = section or ""
        set_argv(f":{section}:{command}")

        result = typeo(
            self.base_func, command1=self.command1, command2=self.command2
        )()

        name = " ".join([v for k, v in command_dict.items() if "name" in k])
        assert self.mock.a == a
        assert self.mock.name == name
        assert result == command_dict["age"] * self.mock.a * a


@pytest.mark.depends(on=["test_config"])
def test_subcommands(subcommands_config_no_sections, fname, set_argv):
    command, command_dict = subcommands_config_no_sections
    SubcommandsTester().test_with_config(
        command, command_dict, fname, set_argv
    )


@pytest.mark.depends(on=["test_config"])
def test_subcommands_with_section(
    subcommands_with_sections_config, fname, set_argv
):
    command, command_dict = subcommands_with_sections_config
    SubcommandsTester().test_with_config(
        command, command_dict, fname, set_argv, "foo"
    )


@pytest.mark.depends(on=["test_config"])
def test_spoof(simple_config_no_fail, fname):
    a, b = simple_config_no_fail

    if fname is None:
        result = spoof(simple_func_with_default)
    else:
        result = spoof(simple_func_with_default, filename=fname)

    assert result["a"] == a
    assert result["b"] == b or "foo"


@pytest.mark.depends(on=["test_config"])
def test_spoof_with_section(simple_config_with_section, fname):
    a, b = simple_config_with_section

    if fname is None:
        result = spoof(simple_func_with_default, script="foo")
    else:
        result = spoof(simple_func_with_default, filename=fname, script="foo")

    assert result["a"] == a
    assert result["b"] == b or "foo"
