from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

from tritonclient.grpc import model_config_pb2 as model_config

from hermes.quiver import Platform

if TYPE_CHECKING:
    from hermes.quiver import Model
    from hermes.quiver.types import SHAPE_TYPE


KIND_TYPE = Literal["auto", "cpu", "gpu"]
GPUS_TYPE = Union[int, Sequence[int], None]


def _add_exposed_tensor(f):
    """
    Decorator for adding input/output adding methods
    to the config class. Doing it this way in order to simplify
    things like syntax updates and building the data type map
    """
    exposed_type = f.__name__.split("_")[1]
    output_type = getattr(model_config, "Model" + exposed_type.title())

    def wrapper(
        obj: "ModelConfig",
        name: str,
        shape: "SHAPE_TYPE",
        dtype: Literal["float32", "int64"] = "float32",
        **kwargs,  # including kwargs for reshaping later or something
    ) -> output_type:
        """Add an {exposed} tensor to the config

        Appends an additional entry to `ModelConfig.{exposed}`
        with the specified keyword arguments.

        Args:
            name:
                The name of the {exposed}
            shape:
                The shape of the {exposed}, with `None`
                representing variable-length axes
            dtype:
                The datatype of the {exposed}
        """.format(
            exposed=exposed_type
        )

        # TODO: Handle datatypes more robustly
        if dtype == "float32":
            dtype = model_config.DataType.TYPE_FP32
        elif dtype == "int64":
            dtype = model_config.DataType.TYPE_INT64
        else:
            raise ValueError(f"Unknown datatype {dtype}")

        shape = (x or -1 for x in shape)
        exposed_obj = output_type(
            name=name,
            dims=shape,
            data_type=dtype,
        )

        current_exposed = getattr(obj._config, exposed_type)
        current_exposed.append(exposed_obj)
        f(exposed_obj, **kwargs)
        return exposed_obj

    wrapper.__name__ = f.__name__
    return wrapper


class ModelConfig:
    """Wrapper around the `tritonclient.grpc.model_config_pb2.ModelConfig`.

    Args:
        model:
            The `Model` object to which this config belongs
        **kwargs:
            Any additional keyword arguments
            with which to initialize the config
    """

    def __new__(cls, model: "Model", **kwargs) -> "ModelConfig":
        if model.platform == Platform.ENSEMBLE:
            cls = EnsembleConfig

        obj = super().__new__(cls)
        obj.__init__(model, **kwargs)
        return obj

    def __init__(self, model: "Model", **kwargs) -> None:
        self.model = model

        # make sure no kwargs were passed that
        # might override the values grabbed
        # from the Model itself
        if "name" in kwargs:
            raise ValueError(
                "Cannot pass 'name' as an argument to ModelConfig"
            )
        elif "platform" in kwargs:
            raise ValueError(
                "Cannot pass 'platform' as an argument to ModelConfig"
            )

        try:
            # try to read an existing config if it exists
            config = model.fs.read_config(
                model.fs.join(model.name, "config.pbtxt")
            )

            # ensure that the name in the config
            # matches the name passed from the model
            if config.name != model.name:
                raise ValueError(
                    "Name in existing config {} "
                    "doesn't match model name {}".format(
                        config.name, model.name
                    )
                )

            # do the same for the platform
            if config.platform != model.platform.value:
                raise ValueError(
                    "Platform in existing config {} "
                    "doesn't match model platform {}".format(
                        config.platform, model.platform.value
                    )
                )

            # add in any kwargs passed to overwrite
            # their existing value in the config
            kwargs_config = model_config.ModelConfig(**kwargs)
            config.MergeFrom(kwargs_config)

        except FileNotFoundError:
            # create a new config if one doesn't
            # already exist
            config = model_config.ModelConfig(
                name=model.name,
                platform=model.platform.value,
                **kwargs,
            )

        # add it as an attribute
        self._config = config

    def __getattr__(self, name):
        """
        Override of `__getattr__` to look for
        attributes directly on the `_config` object
        if they don't exist on this object, e.g.
        `ModelConfig.input` will return the `ModelInput`
        message on the underlying `ModelConfig._config`
        """
        try:
            config = object.__getattribute__(self, "_config")
            return config.__getattribute__(name)
        except AttributeError as e:
            raise AttributeError from e

    def write(self):
        """
        Write out the protobuf config to the model's
        folder in the model repository
        """
        path = self.model.fs.join(self.model.name, "config.pbtxt")
        self.model.fs.write_config(self._config, path)

    @_add_exposed_tensor
    def add_input(input: model_config.ModelInput, **kwargs):
        """
        add an input
        """
        return

    @_add_exposed_tensor
    def add_output(output: model_config.ModelOutput, **kwargs):
        """
        add an output
        """
        return

    def add_instance_group(
        self,
        kind: KIND_TYPE = "gpu",
        gpus: GPUS_TYPE = None,
        count: int = 1,
        name: Optional[str] = None,
    ) -> model_config.ModelInstanceGroup:
        try:
            kind = model_config.ModelInstanceGroup.Kind.Value(
                "KIND_{}".format(kind.upper())
            )
        except ValueError:
            raise ValueError(
                f"Could not understand instance group kind {kind}, "
                "must be one of auto, gpu, cpu"
            )

        if isinstance(gpus, int):
            if gpus < 1:
                raise ValueError(f"Invalid number of gpus specified {gpus}")
            gpus = [i for i in range(gpus)]

        # intialize a new instance group
        instance_group = model_config.ModelInstanceGroup(
            kind=kind, gpus=gpus, count=count, name=name
        )
        self.instance_group.append(instance_group)

        return instance_group

    def scale_instance_group(
        self, count: int, name: Union[str, int, None] = None
    ) -> model_config.ModelInstanceGroup:
        if len(self.instance_group) == 0:
            raise ValueError(
                "Config for model {} has no instance groups "
                "to scale".format(self.name)
            )

        if name is None or isinstance(name, int):
            name = name or 0
            try:
                group = self.instance_group[name]
            except IndexError:
                raise IndexError(
                    "Config for model {} with {} instance groups "
                    "has no instance group at index {}".format(
                        self.name, len(self.instance_group), name
                    )
                )
        elif isinstance(name, str):
            groups = [g.name for g in self.instance_group if g.name == name]
            if len(groups) == 0:
                raise ValueError(
                    "Config for model {} has no instance groups "
                    "named {}".format(self.name, name)
                )
            group = groups[0]
        else:
            raise TypeError(
                "Unexpected type for argument `name` {}".format(type(name))
            )

        group.count = count
        return group

    def __repr__(self):
        return self._config.__repr__()

    def __str__(self):
        return str(self._config)


class EnsembleConfig(ModelConfig):
    def add_step(self, model: "Model", version: Optional[int] = None):
        version = version or -1
        step = model_config.ModelEnsembling.Step(
            model_name=model.name, model_version=version
        )
        self._config.ensemble_scheduling.step.append(step)
        return step
