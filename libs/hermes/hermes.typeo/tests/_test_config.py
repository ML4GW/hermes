import argparse
import os
import shutil
import sys
from typing import Dict, Sequence

import pytest
import toml

from hermes.typeo import typeo
from hermes.typeo.actions import TypeoTomlAction


def set_argv(*args):
    sys.argv = [None] + list(args)


@pytest.fixture
def simple_good_config():
    return {
        "typeo": {
            "a": 1,
            "b": [10, 8, 6],
            "c": {"thom": "vocals", "jonny": "guitar"},
        }
    }


@pytest.fixture(
    scope="module",
    params=[None, os.path.join("tests", "pyproject.toml"), "config.toml"],
)
def simple_config_path(simple_good_config, request):
    if request.param is None:
        shutil.move("pyproject.toml", "_pyproject.toml")
        with open("pyproject.toml", "w") as f:
            toml.dump({"tool": simple_good_config}, f)
        set_argv("--typeo")
        yield request.param
        shutil.move("_pyproject.toml", "pyproject.toml")
        return

    if os.path.basename(request.param) == "pyproject.toml":
        config = {"tool": simple_good_config}
    else:
        config = simple_good_config

    with open(request.param, "w") as f:
        toml.dump(config, f)

    set_argv("--typeo", request.param)
    yield request.param
    os.remove(request.param)


def test_parser_action(simple_config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--typeo", type=str, default=None, action=TypeoTomlAction
    )
    flags = parser.parse_args()

    assert flags.typeo == [
        "--a",
        "1",
        "--b",
        "10",
        "8",
        "6",
        "--c",
        "thom=vocals",
        "jonny=guitar",
    ]


@pytest.mark.depends(on=["test_typeo.py::test_typeo", "test_parser_action"])
def test_config(simple_config_path):
    @typeo
    def func(a: int, b: Sequence[int], c: Dict[str, str]):
        d = {"b": [i + a for i in b]}
        d.update(c)
        return d

    result = func()
    expected = {"b": [11, 9, 7], "thom": "vocals", "jonny": "guitar"}

    for key, val in result.items():
        assert expected.pop(key) == val
