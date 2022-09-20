import logging
import os
import time
from contextlib import contextmanager
from queue import Empty, Queue
from threading import Thread
from typing import Iterable, Optional

from spython.instance import Instance
from spython.main import Client as SingularityClient
from tritonclient import grpc as triton

DEFAULT_IMAGE = (
    "/cvmfs/singularity.opensciencegrid.org/fastml/gwiaas.tritonserver:latest"
)

# TODO: use apps to find this automatically?
TRITON_BINARY = "/opt/tritonserver/bin/tritonserver"


def target(q: Queue, instance: Instance, cmd: str, *args, **kwargs):
    try:
        kwargs["return_result"] = True
        cmd = ["/bin/bash", "-c", cmd]
        response = SingularityClient.execute(instance, cmd, *args, **kwargs)
    except Exception as e:
        msg = f"Failed to execute server command, encountered error {e}"
        response = {"return_code": 1, "message": msg}
    finally:
        q.put(response)


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


def get_wait(q: Queue, log_file: Optional[str] = None):
    def wait(
        endpoint: str = "localhost:8001",
        timeout: Optional[float] = None,
        log_interval: float = 10,
    ) -> None:
        client = triton.InferenceServerClient(endpoint)
        logging.info("Waiting for server to come online")

        timer = Timer(timeout, log_interval)
        live = False
        while timer.tick():
            try:
                live = client.is_server_live()
            except triton.InferenceServerException:
                pass
            finally:
                if live:
                    break

                # the server isn't available yet for some reason
                try:
                    # check if self._thread has finished executing
                    # and placed the response in the _response_queue.
                    # If so, something has gone wrong
                    response = q.get_nowait()
                    if log_file is not None and len(response["message"]) == 0:
                        with open(log_file, "r") as f:
                            response["message"] = f.read()

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

    return wait


@contextmanager
def serve(
    model_repo_dir: str,
    image: str = DEFAULT_IMAGE,
    name: Optional[str] = None,
    gpus: Optional[Iterable[int]] = None,
    server_args: Optional[Iterable[str]] = None,
    log_file: Optional[str] = None,
    wait: bool = False,
) -> Instance:
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
        name:
            Name to give to the Singularity container instance
            in which the server will run.
        gpus:
            The gpu indices to expose to Triton via the
            `CUDA_VISIBLE_DEVICES` environment variable. Note
            that if the host environment has the `CUDA_VISIBLE_DEVICES`
            variable set, the passed indices will be assumed to be
            relative to the GPU ids as specified in that variable.
            For example, if `CUDA_VISIBLE_DEVICES=4,2,6` on the
            host environment, then passing `gpus=[1, 2]` will set
            `CUDA_VISIBLE_DEVICES=2,6` inside the container.
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
        A `spyton.instance.Instance` representing the singularity
            instance running the server, with an additional `wait`
            method included that will hold up the calling thread
            until the server is online.
    """

    # create the base triton server command and
    # point it at the model repository
    cmd = f"{TRITON_BINARY} --model-repository {model_repo_dir}"

    # add in any additional arguments to the server
    if server_args is not None:
        cmd += " " + " ".join(server_args)

    # if we specified a log file, reroute stdout and stderr
    # to that file (triton primarily uses stderr)
    if log_file is not None:
        cmd += f" > {log_file} 2>&1"

    # add environment variables for the specified GPUs
    environ = {}
    if gpus is not None:
        host_visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if host_visible_gpus is not None:
            host_visible_gpus = host_visible_gpus.split(",")

            mapped_gpus = []
            for gpu in gpus:
                try:
                    gpu = host_visible_gpus[gpu]
                except IndexError:
                    raise ValueError(
                        "GPU index {} too large for host environment "
                        "with only {} available GPUs".format(
                            gpu, len(host_visible_gpus)
                        )
                    )
                mapped_gpus.append(gpu)
            gpus = mapped_gpus

        environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

    # spin up a container instance using the specified image
    instance = SingularityClient.instance(
        image,
        name=name,
        start=True,
        quiet=False,  # if we don't set this, the -s doesn't matter
        options=["--nv"],
        singularity_options=["-s"],
        environ=environ,
    )

    # execute the command inside the running container instance.
    # Run it in a separate thread so that we can do work
    # while it runs in this same process
    q = Queue()
    runner = Thread(target=target, args=(q, instance, cmd))
    runner.start()
    instance.wait = get_wait(q, log_file)
    try:
        if wait:
            instance.wait()
        yield instance
    finally:
        logging.debug(f"Stopping container instance {instance.name}")
        instance.stop()

        logging.debug("Waiting for server to shut down")
        runner.join()

        logging.debug("Server container instance successfully spun down")
