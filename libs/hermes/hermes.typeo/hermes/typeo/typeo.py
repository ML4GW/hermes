import argparse
import inspect
from collections import abc
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional, Tuple, TypeVar, Union

from hermes.typeo.actions import CallableAction, EnumAction, MappingAction

if TYPE_CHECKING:
    try:
        from types import GenericAlias
    except ImportError:
        from typing import _GenericAlias as GenericAlias


_LIST_ORIGINS = (list, abc.Sequence)
_ARRAY_ORIGINS = _LIST_ORIGINS + (tuple,)
_DICT_ORIGINS = (dict, abc.Mapping)
_ANNOTATION = Union[type, "GenericAlias"]
_MAYBE_TYPE = Optional[type]


def _parse_union(param: inspect.Parameter) -> Tuple[type, _MAYBE_TYPE]:
    annotation = param.annotation
    if len(annotation.__args__) > 2:
        raise ValueError(
            "Can't parse argument {} with annotation {} "
            "that is a Union of more than 2 types.".format(
                param.name, annotation
            )
        )

    # type_ will be the type that we pass to the
    # parser. second_type can either be `None` or
    # some array of type_
    type_, second_type = annotation.__args__

    try:
        origin = second_type.__origin__
    except AttributeError:
        try:
            # see if the second type in the Union is NoneType
            if isinstance(None, second_type):
                # this is basically a typing.Optional case
                # make sure that the default is None
                if param.default is not None:
                    raise ValueError(
                        "Argument {} with Union of type {} and "
                        "NoneType must have a default of None".format(
                            param.name, type_
                        )
                    )
                return type_, None
        except TypeError:
            # annotation.__args__[1] is not a type that we
            # can check `None` as an instance of, so the
            # `isinstance` check above raises a TypeError.
            pass

        # this annotation isn't NoneType and doesn't have an
        # __origin__, so we can infer that it's not array-like
        # and we don't know how to parse this arg
        raise TypeError(
            "Arg {} has Union of types {} and {}".format(
                param.name, type_, second_type
            )
        )

    def _is_valid(idx=None):
        try:
            args = second_type.__args__
        except AttributeError:
            # in py3.9, generic aliases with no type
            # specified won't have __args__ at all,
            # so this is the same as being TypeVar
            # in py<3.9 and we're ok
            return True

        if idx is not None:
            args = [args[idx]]

        for t in args:
            if not (t in (type_, Ellipsis) or isinstance(t, TypeVar)):
                return False
        return True

    if origin in _LIST_ORIGINS:
        assert _is_valid(0)
    elif origin in _DICT_ORIGINS:
        assert _is_valid(1)
    elif origin is tuple:
        assert _is_valid()
    else:
        raise TypeError(
            "Arg {} has Union of type {} and type {} "
            "with unknown origin {}".format(
                param.name, type_, second_type, origin
            )
        )

    return second_type, type_


def _get_origin_and_type(
    annotation: _ANNOTATION, type_: _MAYBE_TYPE = None
) -> Tuple[_MAYBE_TYPE, _MAYBE_TYPE]:
    """Utility for parsing the origin of an annotation

    Returns:
        If the annotation has an origin, this will be that origin.
            Otherwise it will be `None`
        If the annotation does not have an origin, this will
            be the annotation. Otherwise it will be `None`
    """

    try:
        return annotation.__origin__, type_
    except AttributeError:
        # annotation has no origin, so assume it's
        # a valid type on its own
        return None, annotation


def _parse_array_like(
    annotation: _ANNOTATION, origin: _MAYBE_TYPE, kwargs: dict, type_: type
) -> _MAYBE_TYPE:
    """Make sure array-like typed arguments pass the right type to the parser

    For an annotation with an origin, do some checks on the
    origin to make sure that the type and action argparse
    uses to parse the argument is correct. If the annotation
    doesn't have an origin, returns `None`.

    Args:
        annotation:
            The annotation for the argument
        origin:
            The origin of the annotation, if it exists,
            otherwise `None`
        kwargs:
            The dictionary of keyword arguments to be
            used to add an argument to the parser
    """

    if origin in _ARRAY_ORIGINS + _DICT_ORIGINS:
        kwargs["nargs"] = "+"
        try:
            args = annotation.__args__
        except AttributeError:
            args = None

        if origin in _DICT_ORIGINS:
            kwargs["action"] = MappingAction
            if args is None or isinstance(args[0], TypeVar):
                return type_

            # make sure that the expected type
            # for the dictionary key is string
            # TODO: add kwarg for parsing non-str
            # dictionary keys
            assert args[0] is str

            # the type used to parse the values for
            # the dictionary will be the type passed
            # the parser action
            type_ = args[1]
        else:
            try:
                if args is None or isinstance(args[0], TypeVar):
                    # args being None indicates untyped lists
                    # and tuples in py3.9, and args[0] being
                    # TypeVar indicates untyped lists in py3.8
                    return type_
                type_ = args[0]
            except IndexError:
                # untyped Tuples in py3.8 will have an empty __args__
                return type_

            # for tuples make sure that everything
            # has the same type
            if origin is tuple:
                # TODO: use a custom action to verify the
                # number of arguments and map to a tuple
                try:
                    for arg in annotation.__args__[1:]:
                        if arg is not Ellipsis:
                            assert arg == type_
                except IndexError:
                    # if the Tuple only has one arg, we don't need
                    # to worry about checking everything else
                    pass
        return type_
    elif origin is abc.Callable:
        return origin
    elif origin is not None:
        # this is a type with some unknown origin
        raise TypeError(f"Can't help with arg of type {origin}")
    else:
        return type_


def _parse_help(args: str, arg_name: str) -> str:
    """Find the help string for an argument

    Search through the `Args` section of a function's
    doc string for the lines describing a particular
    argument. Returns the empty string if no description
    is found

    Args:
        args:
            The arguments section of a function docstring.
            Should be formatted like
            ```
            '''
            arg1:
                The description for arg1
            arg2:
                The description for arg 2 that
                spans multiple lines
            arg3:
                Another description
            '''
            ```
            With 8 spaces before each argument name
            and 12 before the lines of its description.
        arg_name:
            The name of the argument whose help string
            to search for
    Returns:
        The help string for the argument with leading
        spaces stripped for each line and newlines
        replaced by spaces
    """

    doc_str, started = "", False
    for line in args.split("\n"):
        # TODO: more robustness on spaces
        if line == (" " * 8 + arg_name + ":"):
            started = True
        elif not line.startswith(" " * 12) and started:
            break
        elif started:
            doc_str += " " + line.strip()
    return doc_str


def make_parser(
    f: Callable,
    prog: Optional[str] = None,
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Build an argument parser for a function

    Builds an `argparse.ArgumentParser` object by using
    the arguments to a function `f`, as well as their
    annotations and docstrings (for help printing).
    The type support for annotations is pretty limited
    and needs more documenting here, but for a better
    idea see the unit tests in `../tests/unit/test_typeo.py`.

    Args:
        f:
            The function to construct a command line
            argument parser for
        prog:
            Passed to the `prog` argument of
            `argparse.ArgumentParser`. If left as `None`,
            `f.__name__` will be used
    Returns:
        The argument parser for the given function
    """

    # start by grabbing the function description
    # and any arguments that might have been
    # described in the docstring
    try:
        # split thet description and the args
        # by the expected argument section header
        doc, args = f.__doc__.split("Args:\n")
    except AttributeError:
        # raised if f doesn't have documentation
        doc, args = "", ""
    except ValueError:
        # raised if f only has a description but
        # no argument documentation. Set `args`
        # to the empty string
        doc, args = f.__doc__, ""
    else:
        # try to strip out any returns from the
        # arguments section by using the expected
        # returns header. If there are None, just
        # keep moving
        try:
            args, _ = args.split("Returns:\n")
        except ValueError:
            pass

    if parser is None:
        # build the parser, using a raw text formatter, so that
        # any formatting in the argument description is respected
        parser = argparse.ArgumentParser(
            prog=prog or f.__name__,
            description=doc.rstrip(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    # now iterate through the arguments of f
    # and add them as options to the parser
    for name, param in inspect.signature(f).parameters.items():
        annotation = param.annotation
        kwargs = {}

        # check to see if the annotation represents
        # a type that can be used by the parser, or
        # represents some container that needs
        # further parsing
        origin, type_ = _get_origin_and_type(annotation)

        # if the annotation can have multiple types,
        # figure out which type to pass to the parser
        if origin is Union:
            annotation, type_ = _parse_union(param)

            # check the chosen type again to
            # see if it's a container of some kind
            origin, type_ = _get_origin_and_type(annotation, type_)

        # if the origin of the annotation is array-like,
        # indicate that there will be multiple args in the kwargs
        # and return the appropriate type. This returns `None`
        # if there's no origin to process, in which case we just
        # keep using `type_`
        type_ = _parse_array_like(annotation, origin, kwargs, type_)

        # our last origin check to see if type_ is typing.Callable,
        # in which case the origin will be abc.Callable whic
        # is the type that we want
        origin, type_ = _get_origin_and_type(type_)
        if origin is not None:
            type_ = origin

        # add the argument docstring to the parser help
        kwargs["help"] = _parse_help(args, name)

        if type_ is bool:
            if param.default is inspect._empty:
                # if the argument is a boolean and doesn't
                # provide a default, assume that setting it
                # as a flag indicates a `True` status
                kwargs["action"] = "store_true"
            else:
                # otherwise set the action to be the
                # _opposite_ of whatever the default is
                # so that if it's not set, the default
                # becomes the values
                action = str(not param.default).lower()
                kwargs["action"] = f"store_{action}"
        else:
            kwargs["type"] = type_

            # args without default are required,
            # otherwise pass the default to the parser
            if param.default is inspect._empty:
                kwargs["required"] = True
            else:
                kwargs["default"] = param.default

            if type_ is abc.Callable:
                kwargs["action"] = CallableAction
            elif type_ is not None and issubclass(type_, Enum):
                kwargs["action"] = EnumAction
                kwargs["choices"] = [i.value for i in type_]

        # use dashes instead of underscores for
        # argument names
        name = name.replace("_", "-")
        parser.add_argument(f"--{name}", **kwargs)
    return parser


def _make_wrapper(
    f: Callable, prog: Optional[str] = None, **kwargs
) -> Callable:
    parser = make_parser(f, prog)
    if len(kwargs) > 0:
        subparsers = parser.add_subparsers(dest="_subprogram", required=True)
        for func_name, func in kwargs.items():
            subparser = subparsers.add_parser(func_name.replace("_", "-"))
            make_parser(func, None, subparser)

    @wraps(f)
    def wrapper(*args, **kw):
        if len(args) == len(kw) == 0:
            kw = vars(parser.parse_args())

        try:
            subprogram = kw.pop("_subprogram")
        except KeyError:
            subprogram = None
        else:
            subprogram = kwargs[subprogram.replace("-", "_")]
            parameters = inspect.signature(subprogram).parameters
            subkw = {name: kw.pop(name) for name in parameters}

        result = f(*args, **kw)
        if subprogram is not None:
            result = subprogram(**subkw)
        return result

    return wrapper


def typeo(*args, **kwargs) -> Callable:
    """Function wrapper for passing command line args to functions

    Builds a command line parser for the arguments
    of a function so that if it is called without
    any arguments, its arguments will be attempted
    to be parsed from `sys.argv`.

    Usage:
        If your file `adder.py` looks like ::

            from hermes.typeo import typeo


            @typeo
            def f(a: int, other_number: int = 1) -> int:
                '''Adds two numbers together

                Longer description of the process of adding
                two numbers together.

                Args:
                    a:
                        The first number to add
                    other_number:
                        The other number to add whose description
                        inexplicably spans multiple lines
                '''

                print(a + other_number)


            if __name__ == "__main__":
                f()

        Then from the command line (note that underscores
        get replaced by dashes!) ::
            $ python adder.py --a 1 --other-number 2
            3
            $ python adder.py --a 4
            5
            $ python adder.py -h
            usage: f [-h] --a A [--other-number OTHER_NUMBER]

            Adds two numbers together

                Longer description of the process of adding
                two numbers together.

            optional arguments:
              -h, --help            show this help message and exit
              --a A                 The first number to add
              --other-number OTHER_NUMBER
                                    The other number to add whose description inexplicably spans multiple lines  # noqa

    Args:
        f:
            The function to expose via a command line parser
        prog:
            The name to assign to command line parser `prog`
            argument. If not provided, `f.__name__` will
            be used.
    """

    # the only argument is the function itself,
    # so just treat this like a simple wrapper
    if len(args) == 1 and isinstance(args[0], Callable):
        return _make_wrapper(args[0], **kwargs)
    else:
        # we provided arguments to typeo above the
        # decorated function, so wrap the wrapper
        # using the provided arguments

        @wraps(typeo)
        def wrapperwrapper(f):
            return _make_wrapper(f, *args, **kwargs)

        return wrapperwrapper
