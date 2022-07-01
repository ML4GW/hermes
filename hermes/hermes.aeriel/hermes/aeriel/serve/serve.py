import logging
from contextlib import contextmanager
from typing import Iterable, Optional

from hermes.aeriel.serve.singularity import SingularityInstance

DEFAULT_IMAGE = (
    "/cvmfs/singularity.opensciencegrid.org/fastml/gwiaas.tritonserver:latest"
)


@contextmanager
def serve(
    model_repo_dir: str,
    image: str = DEFAULT_IMAGE,
    gpus: Optional[Iterable[int]] = None,
    server_args: Optional[Iterable[str]] = None,
    log_file: Optional[str] = None,
    wait: bool = False,
) -> None:
    """Context which spins up a Triton container in the background

    A context for using Singularity to deploy a local
    Triton Inference Server in the background to which
    inference requests can be sent, deployed via Singularity's
    Python APIs in a background thread. If `wait` is `True`,
    the context will not enter until the server is ready
    to receive requests. Otherwise, a `SingularityInstance`
    object will be returned with a `.wait` method which can
    be called at any time to pause until the server is ready.
    At context exit, the Singularity container will be destroyed
    and the server will spin down with it.

    Args:
        model_repo_dir:
            Path to a Triton model repository from which the
            inference service ought to load models
        image:
            The path to the Singularity image to execute
            Triton inside of. Defaults to the image published
            by Hermes to the Open Science Grid.
        gpus:
            The gpu indices to expose to Triton via the
            `CUDA_VISIBLE_DEVICES` environment variable. Note
            that since we use this environment variable, GPU
            indices need to be set relative to their _global_
            value, not to the values mapped to by the value of
            `CUDA_VISIBLE_DEVICES` in the environment from which
            this function is called.
        server_args:
            Additional arguments with which to initialize the
            `tritonserver` executable
        log_file:
            A path to which to pipe Triton's stdout and stderr
            logs. If left as `None`, Triton's logs will not
            be captured.
        wait:
            If `True`, don't enter the context until the inference
            service has come online or returned an error. Otherwise,
            enter the context immediately. The latter might be useful
            if there are other setup tasks that can be done in parallel
            to Triton coming online, or if the user requires more
            fine-grained control over whether to timeout and raise
            an error if Triton takes too long to come online.
    Yields:
        A `SingularityInstance` object representing the container
        instance running Triton.
    """

    # start a container instance from the specified image with
    # the --nv flag set in order to utilize GPUs
    logging.debug(f"Starting instance of singularity image {image}")
    instance = SingularityInstance(image)

    # specify GPUs at the front of the command using
    # CUDA_VISIBLE_DEVICES environment variable
    cmd = ""
    if gpus is not None:
        cmd = "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, gpus)) + " "

    # create the base triton server command and
    # point it at the model repository
    cmd += "/opt/tritonserver/bin/tritonserver "
    cmd += "--model-repository " + model_repo_dir

    # add in any additional arguments to the server
    if server_args is not None:
        cmd += " " + " ".join(server_args)

    # if we specified a log file, reroute stdout and stderr
    # to that file (triton primarily uses stderr)
    if log_file is not None:
        cmd += f" > {log_file} 2>&1"

    # execute the command inside the running container instance.
    # Run it in a separate thread so that we can do work
    # while it runs in this same process
    instance.run(cmd, background=True)
    with instance:
        if wait:
            instance.wait()
        yield instance
