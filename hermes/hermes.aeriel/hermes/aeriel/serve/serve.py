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


class SingularityInstance:
    def __init__(self, image: str = DEFAULT_IMAGE):
        self._instance = SingularityClient.instance(
            image, options=["--nv"], quiet=True
        )
        self._thread = None
        self._response_queue = None

    @property
    def name(self):
        return self._instance.name

    def run(self, command: str, background: bool = False) -> Optional[str]:
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
                self._response_queue.put(response)
            else:
                return response

    def wait(
        self,
        url: str = "localhost:8001",
        timeout: Optional[float] = None,
        log_interval: float = 10,
    ):
        if self._thread is None:
            raise ValueError("No server instance running in background")

        client = triton.InferenceServerClient(url)
        logging.info("Waiting for server to come online")
        start_time, i = time.time(), 1
        while True:
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
                    pass
            finally:
                # keep tabs on how long we've been waiting
                elapsed = time.time() - start_time
                if timeout is not None and elapsed > timeout:
                    raise RuntimeError(
                        f"Server still not online after {timeout}s"
                    )
                elif elapsed >= (i * log_interval):
                    logging.debug(
                        "Still waiting for server to start, "
                        "{}s elapsed".format(i * log_interval)
                    )
                    i += 1
        logging.info("Server online")

    def stop(self):
        logging.debug(f"Stopping singularity instance {self.name}")
        self._instance.stop()
        logging.debug(f"Singularity instance {self.name} stopped")

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.stop()

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
