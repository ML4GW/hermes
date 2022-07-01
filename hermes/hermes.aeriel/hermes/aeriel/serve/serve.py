import logging
import time
from contextlib import contextmanager
from queue import Empty, Queue
from threading import Thread, current_thread
from typing import Iterable, Optional

from spython.main import Client as SingularityClient
from tritonclient import grpc as triton

DEFAULT_IMAGE = (
    "/cvmfs/singularity.opensciencegrid.org/fastml/gwiaas.tritonserver:latest"
)


class Timer:
    def __init__(self, timeout: Optional[float], log_interval: float = 10):
        self.timeout = timeout or float("inf")
        self.log_interval = log_interval
        self._start_time = time.time()
        self._i = 1

    @property
    def current_interval(self):
        return self._i * self.log_interval

    def tick(self):
        elapsed = time.time() - self._start_time
        if elapsed >= self.current_interval:
            logging.debug(
                "Still waiting for server to start, "
                "{}s elapsed".format(self.current_interval)
            )
            self._i += 1
        return elapsed < self.timeout


class SingularityInstance:
    """A running Singularity container instance manager

    Creates a Singularity container instance from the specified
    container image and offers utility methods for executing
    commands inside of it, both in the current thread and
    in the background.

    Args:
        image: Path to the container image to run
    """

    def __init__(self, image: str = DEFAULT_IMAGE):
        self._instance = SingularityClient.instance(
            image, options=["--nv"], quiet=True
        )
        self._thread = None
        self._response_queue = None

    @property
    def name(self):
        return self._instance.name

    # TODO: do we annotate the response object here instead
    # or parse the response content out below, raising an
    # error if the return code is anything but 0?
    def run(self, command: str, background: bool = False) -> Optional[str]:
        """Run a command inside the container instance

        Args:
            command: The command to execute
            background:
                Whether to execute the specified command in the
                current thread and wait for a response, or execute
                it in a background thread to run asynchronously.
                Only one command can run in the background at any
                given time.
        Returns:
            If `background = False`, the command response, containing
            both a return code and the stdout stream from the command,
            are returned. Otherwise `None` is returned.
        Raises:
            ValueError:
                If a command is already executing in the background
                for this container instance.
        """

        # TODO: do some check on whether the underlying instance
        # is still running first
        if background and self._thread is not None:
            raise ValueError(
                "Can't run command '{}' in background, already "
                "executing command in thread '{}'".format(
                    command, self._thread.native_id
                )
            )
        elif background:
            self._thread = Thread(target=self.run, args=(command,))
            self._response_queue = Queue()

            logging.debug(
                f"Executing command on thread {self._thread.native_id}"
            )
            self._thread.start()
        else:
            logging.debug(
                "Executing command '{}' in singularity instance {}".format(
                    command, self.name
                )
            )
            command = ["/bin/bash", "-c", command]
            response = SingularityClient.execute(self._instance, command)

            # check if this is being called from the separate
            # thread so that we can route its response back
            # to the main thread via the response queue
            if self._thread is not None and current_thread() is self._thread:
                # TODO: make a simple way to read from the _response_queue
                self._response_queue.put(response)
            else:
                return response

    def wait(
        self,
        address: str = "localhost:8001",
        timeout: Optional[float] = None,
        log_interval: float = 10,
    ):
        """
        Wait for a Triton inference server to come online
        at the specified address. Requires that a command
        be currently running inside this container instance
        in the background, since it is assumed that this command
        started the server and that if the command has finished
        before the server is online, something has gone wrong.

        Args:
            address: The full address at which to query the server
            timeout:
                How long to wait for the server to be online,
                in seconds, before raising a `RuntimeError`.
                If left as `None`, will wait indefinitely.
            log_interval:
                Time between `DEBUG` level logs indicating that
                this process is alive but is still waiting.
        Raises:
            ValueError:
                If this instance's background command finishes
                executing before the server comes online.
            RuntimeError:
                If `timeout` is not `None` and the server fails
                to come online before the timeout deadline is reached.
        """
        if self._thread is None:
            raise ValueError("No server instance running in background")
        client = triton.InferenceServerClient(address)
        logging.info("Waiting for server to come online")

        timer = Timer(timeout, log_interval)
        while timer.tick():
            try:
                if client.is_server_live():
                    # if the server is live, we can get started
                    break
            except triton.InferenceServerException:
                # the server isn't available yet for some reason
                try:
                    # check if self._thread has finished executing
                    # and placed the response in the _response_queue.
                    # If so, something has gone wrong
                    response = self._response_queue.get_nowait()
                    raise ValueError(
                        "Server failed to start with return code "
                        "{return_code} and message:\n{message}".format(
                            **response
                        )
                    )
                except Empty:
                    # otherwise we're still just waiting for the
                    # server to come online, so keep waiting
                    continue
        else:
            # the loop above never broke, so we must have timed out
            # TODO: is there a more specific TimeoutError we can call
            raise RuntimeError(f"Server still not online after {timeout}s")
        logging.info("Server online")

    def stop(self):
        """Stop the container instance"""
        logging.debug(f"Stopping singularity instance {self.name}")
        self._instance.stop()
        logging.debug(f"Singularity instance {self.name} stopped")

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        """Stop the instance and join any underlying background thread"""
        self.stop()

        # TODO: should there be a `wait` kwarg at __init__ indicating
        # whether exiting the context should join the thread first?
        # Probably doesn't make sense in the context of Triton which
        # will never join if left to its own devices, but worth keeping
        # in mind for any extensions down the line.
        if self._thread is not None:
            logging.debug(f"Joining thread {self._thread.native_id}")
            self._thread.join()

        self._thread = None
        self._response_queue = None


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
