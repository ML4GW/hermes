# Typeo
## (Pronounced like the abbreviation of "typographical error")
Turn annotated functions into command line scripts with a single line of code, and keep all your documentation!

Uses type annotations on functions to parse command line arguments, and strips help strings from function documentation.


## Basic Usage

Say we have a file `say_hello.py` that looks like:
```python
def say_hello(name: str, friendliness: int):
    """Say hello with various degrees of friendliness

    Args:
        name:
            The name of the person to greet
        friendliness:
            The level of friendliness to greet them with
    """

    if friendliness == 0:
        print("hey.")
    elif friendliness == 1:
        print(f"Hi {name}")
    elif friendliness > 1:
        print(f"Hello {name}!")
    else:
        raise ValueError(
            "Friendliness level cannot be less than 0"
        )
```

This function can be run as a command line utility by just adding

```python
from hermes.typeo import typeo


@typeo
def say_hello(name: str, friendliness: int = 1):
    """Say hello to someone with various degrees of friendliness

    Args:
        name:
            The name of the person to greet
        friendliness:
            The level of friendliness to greet them with
    """

    if friendliness == 0:
        print("hey.")
    elif friendliness == 1:
        print(f"Hi {name}")
    elif friendliness > 1:
        print(f"Hello {name}!")
    else:
        raise ValueError(
            "Friendliness level cannot be less than 0"
        )


if __name__ == "__main__":
    say_hello()
```

Now when we run from the command line:
```console
$ python say_hello.py -h
usage: say_hello [-h] --name NAME [--friendliness FRIENDLINESS]

Say hello to someone with various degrees of friendliness

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           The name of the person to greet (default: None)
  --friendliness FRIENDLINESS
                        The level of friendliness to greet them with (default: 1)

$ python say_hello.py --name Thom
Hi Thom

$ python say_hello.py --name Thom --friendliness 2
Hello Thom!

$ python say_hello.py --friendliness 0
usage: say_hello [-h] --name NAME [--friendliness FRIENDLINESS]
say_hello: error: the following arguments are required: --name
```

Note that we can still import `say_hello` in other scripts and call it with regular arguments, and its behavior won't be impacted. `typeo` works by only reading from the command line if no arguments are passed in.

```python
from say_hello import say_hello


# prints "Hi Thom"
say_hello("Thom", 1)
```

Note also that we can drop the `if __name__ == "__main__"` syntax if we integrate with Poetry package scripts. For example, if my `pyproject.toml` in the directory where I host `say_hello.py` has a section like:

```toml
[tool.poetry.scripts]
greet = "say_hello:say_hello"
```

Then I can drop the `if __name__ == "__main__"` from `say_hello.py` and run my script like this

```console
$ poetry run greet --name Thom
Hi Thom
```

## Subcommands

We can also add subcommands to our scripts. Let's say `greet.py` looks like

```python
from hermes.typeo import typeo


def validate_friendliness(friendliness: int):
    if friendliness < 0:
        raise ValueError(
            "Friendliness level cannot be less than 0"
        )


def say_goodbye(name: str, friendliness: int = 1):
    """Say goodbye to someone with various degrees of friendliness

    Args:
        name:
            The name of the person to bid farewell
        friendliness:
            The level of friendliness to bid them farewell with
    """

    validate_friendliness(friendliness)

    if friendliness == 0:
        print("bye.")
    elif friendliness == 1:
        print(f"Goodbye {name}")
    else:
        print(f"So long {name}!")


def say_hello(name: str, friendliness: int = 1):
    """Say hello to someone with various degrees of friendliness

    Args:
        name:
            The name of the person to greet
        friendliness:
            The level of friendliness to greet them with
    """

    validate_friendliness(friendliness)
    if friendliness == 0:
        print("hey.")
    elif friendliness == 1:
        print(f"Hi {name}")
    else:
        print(f"Hello {name}!")


@typeo(hello=say_hello, goodbye=say_goodbye)
def greet(greeter: str):
    print(f"This is a greeting from {greeter}:")


if __name__ == "__main__":
    greet()
```

```console
$ python greet.py -h
usage: greet [-h] --greeter GREETER {hello,goodbye} ...

positional arguments:
  {hello,goodbye}

optional arguments:
  -h, --help         show this help message and exit
  --greeter GREETER

$ python greet.py hello -h
usage: greet hello [-h] --name NAME [--friendliness FRIENDLINESS]

Say hello to someone with various degrees of friendliness



optional arguments:
  -h, --help            show this help message and exit
  --name NAME           The name of the person to greet (default: None)
  --friendliness FRIENDLINESS
                        The level of friendliness to greet them with (default: 1)

$ python greet.py goodbye -h
usage: greet goodbye [-h] --name NAME [--friendliness FRIENDLINESS]

Say goodbye to someone with various degrees of friendliness



optional arguments:
  -h, --help            show this help message and exit
  --name NAME           The name of the person to bid farewell (default: None)
  --friendliness FRIENDLINESS
                        The level of friendliness to bid them farewell with (default: 1)

$ python greet.py --greeter Jonny hello --name Thom
This is a greeting from Jonny:
Hi Thom

$ python greet.py --greeter Phil goodbye --name Jonny --friendliness 2
This is a greeting from Phil:
So long Jonny!
```
