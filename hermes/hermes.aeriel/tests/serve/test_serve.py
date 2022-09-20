import os
import time
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from hermes.aeriel.serve.serve import get_wait, serve


def test_get_wait():
    q = Queue()
    wait = get_wait(q)
    responses = iter([False, False, True])

    def is_server_live(obj):
        time.sleep(0.1)
        return next(responses)

    with patch(
        "tritonclient.grpc.InferenceServerClient.is_server_live",
        new=is_server_live,
    ):
        wait(timeout=None)

    responses = iter([False, False, True])
    with patch(
        "tritonclient.grpc.InferenceServerClient.is_server_live",
        new=is_server_live,
    ):
        with pytest.raises(RuntimeError):
            wait(timeout=0.1)


@pytest.fixture(params=[None, [0], [3, 4]])
def gpus(request):
    return request.param


@pytest.fixture(params=[None, "log.txt"])
def log_file(request):
    return request.param


@pytest.fixture(params=[None, "--model-control-mode explicit"])
def server_args(request):
    return request.param


@pytest.fixture(params=[None, "test"])
def container_name(request):
    return request.param


def execute(instance, cmd, *args, **kwargs):
    time.sleep(0.1)
    return cmd[-1]


def test_singularity_instance_run(gpus, log_file, server_args, container_name):
    args = None if server_args is None else server_args.split()
    ctx = serve(
        "/path/to/repo",
        "/path/to/image",
        name=container_name,
        gpus=gpus,
        server_args=args,
        log_file=log_file,
    )

    if gpus is None:
        expected_environ = {}
    else:
        devices = ",".join(map(str, gpus))
        expected_environ = {"CUDA_VISIBLE_DEVICES": devices}

    instance_mock = MagicMock()
    with patch(
        "spython.main.Client.instance", return_value=instance_mock
    ) as instance_patch:
        with patch("spython.main.Client.execute") as execute_patch:
            with ctx as instance:
                # make sure that the expected instance gets created
                instance_patch.assert_called_with(
                    "/path/to/image",
                    name=container_name,
                    start=True,
                    quiet=False,
                    options=["--nv"],
                    singularity_options=["-s"],
                    environ=expected_environ,
                )

                # now make sure the wait method on the returned
                # Instance instance behaves reasonably
                with patch(
                    "tritonclient.grpc.InferenceServerClient.is_server_live",
                    return_value=True,
                ):
                    instance.wait()

            # now verify that the appropriate command gets called
            # inside the instance
            cmd = "/opt/tritonserver/bin/tritonserver "
            cmd += "--model-repository /path/to/repo"

            # TODO: this is doing a bit too much of recreating
            # the internal logic for my taste. How can this
            # be made more robust?
            if server_args is not None:
                cmd += " " + server_args
            if log_file is not None:
                cmd += f" > {log_file} 2>&1"
            cmd = ["/bin/bash", "-c", cmd]

            execute_patch.assert_called_with(
                instance_mock, cmd, return_result=True
            )


def test_singularity_instance_run_with_host_devices(gpus):
    visible_devices = list(map(str, [4, 3, 5, 2, 6]))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)

    ctx = serve("/path/to/repo", "/path/to/image", name="test", gpus=gpus)

    if gpus is None:
        expected_environ = {}
    else:
        devices = [visible_devices[i] for i in gpus]
        devices = ",".join(devices)
        expected_environ = {"CUDA_VISIBLE_DEVICES": devices}

    with patch("spython.main.Client.instance") as mock:
        with patch("spython.main.Client.execute"):
            with ctx:
                mock.assert_called_with(
                    "/path/to/image",
                    name="test",
                    start=True,
                    quiet=False,
                    options=["--nv"],
                    singularity_options=["-s"],
                    environ=expected_environ,
                )
