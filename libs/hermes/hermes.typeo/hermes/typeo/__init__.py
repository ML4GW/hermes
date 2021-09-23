import argparse
import inspect
from collections import abc
from functools import wraps
from typing import Callable, Optional, Tuple, Union


class DictParsingAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        self._type = kwargs["type"]
        kwargs["type"] = str
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 1 and "=" not in values[0]:
            setattr(namespace, self.dest, self._type(values[0]))
            return

        dict_value = {}
        for value in values:
            k, v = value.split("=")
            dict_value[k] = self._type(v)
        setattr(namespace, self.dest, dict_value)


def _enforce_array_like(annotation, type_, name):
    annotation = annotation.__args__[1]

    # for array-like origins, make sure that
    # the type of the elements of the array
    # matches the type of the first element
    # of the Union. Otherwise, we don't know
    # how to parse the value
    try:
        if annotation.__origin__ in (list, abc.Sequence):
            assert annotation.__args__[0] is type_
        elif annotation.__origin__ is dict:
            assert annotation.__args__[1] is type_
        elif annotation.__origin__ is tuple:
            for arg in annotation.__args__:
                assert arg is type_
        else:
            raise TypeError(
                "Arg {} has Union of type {} and type {} "
                "with unknown origin {}".format(
                    name, type_, annotation, annotation.__origin__
                )
            )
    except AttributeError:
        raise TypeError(
            "Arg {} has Union of types {} and {}".format(
                name, type_, annotation
            )
        )
    return annotation


def _parse_union(param):
    annotation = param.annotation
    type_ = annotation.__args__[0]

    try:
        if isinstance(None, annotation.__args__[1]):
            # this is basically a typing.Optional case
            # make sure that the default is None
            if param.default is not None:
                raise ValueError(
                    "Argument {} with Union of type {} and "
                    "NoneType must have a default of None".format(
                        param.name, type_
                    )
                )
            annotation = type_
        else:
            annotation = _enforce_array_like(annotation, type_, param.name)
    except TypeError as e:
        if "Subscripted" in str(e):
            annotation = _enforce_array_like(annotation, type_, param.name)
        else:
            raise

    return annotation


def _get_origin_and_type(annotation) -> Tuple[Optional[type], Optional[type]]:
    """Utility for parsing the origin of an annotation

    Returns:
        If the annotation has an origin, this will be that origin.
            Otherwise it will be `None`
        If the annotation does not have an origin, this will
            be the annotation. Otherwise it will be `None`
    """

    try:
        origin = annotation.__origin__
        type_ = None
    except AttributeError:
        # annotation has no origin, so assume it's
        # a valid type on its own
        origin = None
        type_ = annotation
    return origin, type_


def _parse_array_like(annotation, origin, kwargs):
    # use multi-arg handling for arguments
    # that are array-like
    # check for both for <3.8 robustness
    # TODO: do we use a custom action to handle tuples?
    # TODO: what am I missing?
    if origin in (list, tuple, dict, abc.Sequence):
        kwargs["nargs"] = "+"
        if origin is dict:
            kwargs["action"] = DictParsingAction

            # make sure that the expected type
            # for the dictionary key is string
            # TODO: add kwarg for parsing non-int
            # dictionary keys
            assert annotation.__args__[0] is str

            # the type used to parse the values for
            # the dictionary will be the type passed
            # the parser action
            type_ = annotation.__args__[1]
        else:
            type_ = annotation.__args__[0]

            # for tuples make sure that everything
            # has the same type
            if origin is tuple:
                for arg in annotation.__args__[1:]:
                    if arg is not Ellipsis:
                        assert arg == type_

        return type_
    elif origin is not None:
        # this is a type with some unknown origin
        raise TypeError(f"Can't help with arg of type {origin}")


def _parse_help(args, name):
    # search through the docstring lines to get
    # the help string for this argument
    doc_str, started = "", False
    for line in args.split("\n"):
        # TODO: more robustness on spaces
        if line == (" " * 8 + name + ":"):
            started = True
        elif not line.startswith(" " * 12) and started:
            break
        elif started:
            doc_str += " " + line.strip()
    return doc_str


def make_parser(f: Callable, prog: str = None):
    try:
        doc, args = f.__doc__.split("Args:\n")
    except AttributeError:
        doc, args = "", ""
    except ValueError:
        doc, args = f.__doc__, ""
    else:
        try:
            args, _ = args.split("Returns:\n")
        except ValueError:
            pass

    parser = argparse.ArgumentParser(
        prog=prog or f.__name__,
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

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
            annotation = _parse_union(param)

            # check the chosen type again to
            # see if it's a container of some kind
            origin, type_ = _get_origin_and_type(annotation)

        type_ = _parse_array_like(annotation, origin, kwargs) or type_
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

        name = name.replace("_", "-")
        parser.add_argument(f"--{name}", **kwargs)
    return parser


def typeo(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], Callable):
        f = args[0]
        parser = make_parser(f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            if len(args) == len(kwargs) == 0:
                kwargs = vars(parser.parse_args())
            return f(*args, **kwargs)

        return wrapper
    else:

        @wraps(typeo)
        def wrapperwrapper(f):
            parser = make_parser(f, *args, **kwargs)

            @wraps(f)
            def wrapper(*args, **kwargs):
                if len(args) == len(kwargs) == 0:
                    kwargs = vars(parser.parse_args())
                return f(*args, **kwargs)

            return wrapper

        return wrapperwrapper
