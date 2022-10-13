import re
from pathlib import Path
from typing import Any, Union

from hermes.quiver import Model, Platform, io


class ModelRepository:
    """Object representing a directory structured as a Triton model repository

    Triton model repositories are local or cloud-based
    directories that follow a very strict structure. This
    utility class aids in implementing and maintaining
    such a structure and keeping track of the models contained
    in this repository. Currently local directories and
    Google Cloud Storage buckets are supported locations for
    model repositories.

    If a repository already exists at the specified location,
    an attempt will be made to add any existing models to
    the in-memory model cache. If no config file exists for
    these models, their corresponding Platform cannot be
    inferred and so a `ValueError` will be raised.

    Args:
        root:
            The path to the root level of the model repository.
            To specify a Google Cloud Storage bucket, pass a
            string beginning with `"gs://"`.
        clean:
            Whether to remove all existing models from the
            repository. This is useful if previous attempts
            to export models resulted in errors and as a result
            they don't have a valid config.
        kwargs:
            Any keyword arguments to pass to the initialization
            of the underlying filesystem object managing the repository.
    """

    def __init__(
        self, root: Union[str, Path], clean: bool = False, **kwargs: Any
    ) -> None:
        # initialize the filesystem backend for managing
        # reading/writing/path operations
        if isinstance(root, str) and root.startswith("gs://"):
            self.fs = io.GCSFileSystem(root.replace("gs://", ""), **kwargs)
        else:
            self.fs = io.LocalFileSystem(root)

        if clean:
            self.fs.remove("*")
        self.refresh()

    def refresh(self) -> None:
        """
        Reload the in-memory model cache using the current
        state of the repository filesystem.
        """

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
                    "Failed to initialize repo at {} due to "
                    "model {} with missing config.".format(self.fs, model)
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
                    "model {} with unknown platform {}.".format(
                        self.fs, model, config.platform
                    )
                )
            self.add(model, platform)

    @property
    def models(self) -> dict:
        """Return a mapping from model names to `Model` objects."""

        return {model.name: model for model in self._models}

    def add(self, name: str, platform: Platform, force: bool = False) -> Model:
        """
        Add a new model entry to this repository. If the model
        already exists, a `ValueError` will be raised unless
        `force` is `True`, in which case a new model will be
        created with an incremented integer appended to the
        model name.

        Args:
            name:
                Name of the model. Will also be the name of the
                corresponding directory in the model repository.
            platform:
                Desired inference platform Triton will use to
                execute the model at inference time.
            force:
                Whether to force the addition of this model, even
                if a model of the same name already exists in the
                repository. In this case, the model name will be
                appended by `_<idx>`, where `idx` represents the
                lowest non-negative integer possible in order to
                create a unique name in the repository's index.
        Returns:
            The created `Model` object
        """

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

    def remove(self, model: Union[str, Model]):
        if isinstance(model, str):
            try:
                model = [i for i in self._models if i.name == model][0]
            except IndexError:
                raise ValueError(f"Unrecognized model {model}")

        self._models.remove(model)
        self.fs.remove(model.name)

    def delete(self):
        for model in self._models:
            self.remove(model)
        self.fs.delete()
