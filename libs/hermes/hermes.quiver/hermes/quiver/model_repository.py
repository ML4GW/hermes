import re

from gravswell.quiver import Model, Platform, io


class ModelRepository:
    def __init__(self, root: str, **kwargs) -> None:
        # initialize the filesystem backend for managing
        # reading/writing/path operations
        if root.startswith("gs://"):
            self.fs = io.GCSFileSystem(root.replace("gs://", ""), **kwargs)
        else:
            self.fs = io.LocalFileSystem(root)

        # load in any models that already exist in the repository
        # initialize a `_models` attribute to wrap with a `models` property
        self._models = []
        for model in self.fs.list(""):
            # try to read the config for the model, and if it
            # doesn't exist raise an error
            # TODO: better classmethods on Configs and models to
            # load from existing repos and handle these sorts of
            # issues at that level
            try:
                config = self.fs.read_config(
                    self.fs.join(model, "config.pbtxt")
                )
            except FileNotFoundError:
                raise ValueError(
                    f"Failed to initialize repo at {root}"
                    f"due to model with missing config {model}"
                )

            # make sure the specified platform is legitimate
            # note that this is different than whether this is
            # a platform that we can export to: we'll catch that
            # if and when we try to export to it. This just validates
            # that this config specifies a real, Triton-accepted config
            try:
                platform = Platform(config.platform)
            except KeyError:
                raise ValueError(
                    "Failed to initialize repo at {} due to "
                    "model {} with unknown platform {}".format(
                        self.root, model, config.platform
                    )
                )
            self.add(model, platform)

    @property
    def models(self) -> dict:
        """Return a mapping from model names to `Model` objects."""

        return {model.name: model for model in self._models}

    def add(self, name: str, platform: Platform, force: bool = False) -> Model:
        if name in self.models and not force:
            raise ValueError("Model {} already exists".format(name))
        elif name in self.models:
            # append an index to the name of the model starting at 0
            pattern = re.compile(f"{name}_[0-9]+")
            matches = list(filter(pattern.fullmatch, self.models))

            if len(matches) == 0:
                # no postfixed models have been made yet, start at 0
                idx = 0
            else:
                # search for the first available index

                # start by finding all the existing postfixes in use
                pattern = re.compile(f"(?<={name}_)[0-9]+$")
                postfixes = [int(pattern.search(x).group(0)) for x in matches]

                # then sort the postfixes and start
                # counting up, waiting for the first
                # postfix that doesn't match the index,
                # indicating that a spot is available
                for idx, postfix in enumerate(sorted(postfixes)):
                    if postfix != idx:
                        break
                else:
                    # the postfixes are all taken in order,
                    # so increment to take the next available
                    idx += 1
            name += f"_{idx}"

        model = Model(name=name, repository=self, platform=platform)
        self._models.append(model)
        return model

    def remove(self, model_name: str):
        try:
            model = self.models.pop(model_name)
        except KeyError:
            raise ValueError(f"Unrecognized model {model_name}")
        self.fs.remove(model.name)

    def delete(self):
        model_names = self.models.keys()
        for model_name in model_names:
            self.remove(model_name)
        self.fs.delete()
