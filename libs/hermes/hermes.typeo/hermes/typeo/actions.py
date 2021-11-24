import argparse
import importlib
from typing import Callable, Mapping, Optional

import toml


class MappingAction(argparse.Action):
    """Action for parsing dictionary arguments

    Parse dictionary arguments using the form `key=value`,
    with the `type` argument specifying the type of `value`.
    The type of `key` must be a string. Alternatively, if
    a single argument is passed without `=` in it, it will
    be set as the value of the flag using `type`.

    Example ::

        parser = argparse.ArgumentParser()
        parser.add_argument("--a", type=int, action=_DictParsingAction)
        args = parser.parse_args(["--a", "foo=1", "bar=2"])
        assert args.a["foo"] == 1
    """

    def __init__(self, *args, **kwargs) -> None:
        self._type = kwargs["type"]
        kwargs["type"] = str
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if len(values) == 1 and "=" not in values[0]:
            setattr(namespace, self.dest, self._type(values[0]))
            return

        dict_value = {}
        for value in values:
            try:
                k, v = value.split("=")
            except ValueError:
                raise argparse.ArgumentError(
                    self,
                    "Couldn't parse value {} passed to "
                    "argument {}".format(value, self.dest),
                )

            # TODO: introduce try-catch here
            dict_value[k] = self._type(v)
        setattr(namespace, self.dest, dict_value)


class EnumAction(argparse.Action):
    def __init__(self, *args, **kwargs) -> None:
        self._type = kwargs["type"]
        kwargs["type"] = str
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if self.nargs == "+":
            value = []
            for v in values:
                value.append(self._type(v))
        else:
            value = self._type(values)

        setattr(namespace, self.dest, value)


class CallableAction(argparse.Action):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["type"] = str
        super().__init__(*args, **kwargs)

    def _import_callable(self, callable: str) -> Callable:
        fn, module = callable[::-1].split(".", maxsplit=1)
        module, fn = module[::-1], fn[::-1]

        try:
            lib = importlib.import_module(module)
        except ModuleNotFoundError:
            raise argparse.ArgumentError(
                self,
                "Could not find module {} for callable argument {}".format(
                    module, self.dest
                ),
            )

        try:
            # TODO: add inspection of function to make sure it
            # aligns with the Callable __args__ if there are any
            return getattr(lib, fn)
        except AttributeError:
            raise argparse.ArgumentError(
                self,
                "Module {} has no function {} for callable argument {}".format(
                    module, fn, self.dest
                ),
            )

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if self.nargs == "+":
            value = []
            for v in values:
                value.append(self._import_callable(v))
        else:
            value = self._import_callable(values)

        setattr(namespace, self.dest, value)


class TypeoTomlAction(argparse.Action):
    def __init__(
        self, *args, bools: Optional[Mapping[str, bool]] = None, **kwargs
    ) -> None:
        self.bools = bools
        assert kwargs["nargs"] == "?"
        super().__init__(*args, **kwargs)

    def _parse_section(self, section):
        args = ""
        for arg, value in section.items():
            bool_default = None
            if self.bools is not None:
                try:
                    bool_default = self.bools[arg]
                except KeyError:
                    pass

            if bool_default is None and isinstance(value, bool):
                raise argparse.ArgumentError(
                    self,
                    "Can't parse non-boolean argument "
                    "'{}' with value {}".format(arg, value),
                )
            elif bool_default is not None and bool_default == value:
                continue

            args += "--" + arg.replace("_", "-") + " "
            if isinstance(value, bool):
                continue
            elif isinstance(value, dict):
                for k, v in value.items():
                    args += f"{k}={v} "
            elif isinstance(value, list):
                args += " ".join(map(str, value)) + " "
            else:
                args += str(value) + " "
        return args

    def __call__(self, parser, namespace, value, option_string=None):
        if value is None:
            value = "pyproject.toml"

        try:
            filename, command = value.split("::")
        except ValueError:
            if value.startswith("::"):
                command = value.strip(":")
                filename = "pyproject.toml"
            else:
                filename = value
                command = None

        with open(filename, "r") as f:
            config = toml.load(f)

        try:
            config = config["typeo"]
        except KeyError:
            pass

        try:
            commands = config.pop("commands")[command]
        except KeyError:
            commands = None

        args = self._parse_section(config)
        if command is not None:
            args += command + " "
            if commands is not None:
                args += self._parse_section(commands)

        setattr(namespace, self.dest, args.split())
