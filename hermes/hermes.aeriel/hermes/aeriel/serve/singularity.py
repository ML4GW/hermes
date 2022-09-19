import logging
import time
from queue import Empty, Queue
from threading import Thread, current_thread
from typing import Optional

from spython.main import Client as SingularityClient
from tritonclient import grpc as triton


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

    def __init__(self, image):
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
            response = SingularityClient.execute(
                self._instance, command, singularity_options=["-s"]
            )

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
