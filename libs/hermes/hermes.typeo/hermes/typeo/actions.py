import argparse
import importlib
import os
import re
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
        self.base_regex = re.compile(r"(?<!\\)\$\{base.(\w+)\}")
        self.env_regex = re.compile(r"(?<!\\)\$\{(\w+)\}")
        super().__init__(*args, **kwargs)

    def _parse_value(self, value):
        # check if the value is formatted in such a way
        # as to indicate either an environment variable
        # or typeo base-section wildcard by being formatted
        # as ${} (with a \ escaping the $ if it's there)
        value = str(value)
        base_match = self.base_regex.search(value)
        if base_match is not None:
            varname = base_match.group(1)
            try:
                replace = str(self.base[varname])
            except KeyError:
                raise ValueError(
                    "No variable {} indicated in typeo config value {} "
                    "found in base section of typeo config".format(
                        varname, value
                    )
                )
            value = self.base_regex.sub(replace, value)
        else:
            env_match = self.env_regex.search(value)
            if env_match is not None:
                varname = env_match.group(1)
                try:
                    replace = os.environ[varname]
                except KeyError:
                    raise ValueError(
                        "No environment variable {}, referenced "
                        "in typeo config value {}".format(varname, value)
                    )
                value = self.env_regex.sub(replace, value)
        return value

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
                    v = self._parse_value(v)
                    args += f"{k}={v} "
            elif isinstance(value, list):
                args += " ".join(map(self._parse_value, value)) + " "
            else:
                args += self._parse_value(value) + " "
        return args

    def _get_sections(self, config, section, command, filename):
        try:
            # check to see if there are any script-specific
            # config sections at all
            scripts = config.pop("scripts")
        except KeyError:
            scripts = None
            if section is not None and command is not None:
                # if we specified both a script _and_ a command for
                # that script, but there's no script section
                # to pull commands from, we don't know where to
                # get our args from, so I think all we can do is
                # raise an error here
                raise argparse.ArgumentError(
                    self,
                    "Specified script '{}' and command '{}', but"
                    "no 'script' table in config file '{}'".format(
                        section, command, filename
                    ),
                )
            elif command is not None:
                # if we specified a command and not a script,
                # then see if there are args associated with
                # that command at the `typeo` level of the config
                try:
                    commands = config.pop("commands")[command]
                except KeyError:
                    commands = None
            else:
                commands = None
        else:
            # if we do have a `scripts` section of the config,
            # see if we have args associated with the script
            # we passed at the command line
            try:
                scripts = scripts[section]
            except KeyError:
                # if not, then there will be no script-specific
                # args passed _or_ args passed to the indicated
                # command (which could be fine if the command can
                # run without any args. That will be decided by
                # the actual typeo parser at parse time)
                scripts, commands = None, None
            else:
                # see if we have any command-specific arguments
                # for the indicated script
                try:
                    commands = scripts.pop("commands")[command]
                except KeyError:
                    commands = None
        return scripts, commands

    def __call__(self, parser, namespace, value, option_string=None):
        if value is None:
            value = "pyproject.toml"

        # allow specification of the form filename(:script)(:command),
        # where `script` and `command` are optional and
        # - `script` specifies a sub-table of the `typeo` table
        #   in the config file that has arguments specific to
        #   a particular script that's part of the project. A
        #   blank argument here (i.e. either `filename` or
        #   `filename::comand`) will only pull arguments from
        #   the `typeo` section of the config
        # - `command` indicates a particular command for
        #   the indicated script
        try:
            filename, section, command = value.split(":")
        except ValueError:
            # we have at most one colon in `value`, so
            # command is definitely blank
            command = None

            # try a single split to see if we specified
            # a script subsection of the config
            try:
                filename, section = value.split(":")
            except ValueError:
                # no colons at all, so just use the filename
                filename = value
                section = None
        else:
            # if section is the empty string, we have a
            # filename::command, so there's no script subsection
            section = section or None

        if filename == "":
            # if filename is now the empty string, we indicated
            # a subsection and/or a command, but no file. So
            # default to using pyproject.toml
            filename = "pyproject.toml"
        elif os.path.isdir(filename):
            # if `filename` is a directory, look for a
            # pyproject.toml at that location
            filename = os.path.join(filename, "pyproject.toml")

        try:
            # try to load the config file
            with open(filename, "r") as f:
                config = toml.load(f)
        except FileNotFoundError:
            dirname = os.path.dirname(filename) or "."
            basename = os.path.basename(filename)
            raise argparse.ArgumentError(
                self,
                "Could not find typeo config file {} in directory {}".format(
                    basename, dirname
                ),
            )

        if os.path.basename(filename) == "pyproject.toml":
            # if the config file is a pyproject.toml from
            # anywhere, assume that the file uses the
            # standard that all tool configs fall in a
            # `tool` table in the config file
            config = config["tool"]

        # now grab the typeo-specific config
        try:
            config = config["typeo"]
        except KeyError:
            raise argparse.ArgumentError(
                self, f"No 'typeo' section in config file {filename}"
            )

        try:
            self.base = config.pop("base")
        except KeyError:
            self.base = {}

        scripts, commands = self._get_sections(
            config, section, command, filename
        )

        # start by parsing the root typeo-level config options
        args = self._parse_section(config)

        if scripts is not None:
            # if there are any script-specific args to parse,
            # parse them out of the corresponding section
            args += self._parse_section(scripts)

        if command is not None:
            # we specified a command, so add it as a positional
            # argument _after_ all the global arguments
            args += command + " "
            if commands is not None:
                # add in any command-specific arguments after the
                # command argument has been specified
                args += self._parse_section(commands)

        setattr(namespace, self.dest, args.split())
