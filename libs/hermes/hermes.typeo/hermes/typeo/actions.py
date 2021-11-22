import argparse
import importlib
from typing import Callable


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
