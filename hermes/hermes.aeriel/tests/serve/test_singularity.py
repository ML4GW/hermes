import time
from unittest.mock import patch

import pytest

from hermes.aeriel.serve.singularity import SingularityInstance


@patch("spython.main.Client.instance")
@patch("spython.main.Client.execute", return_value=True)
def test_singularity_instance_run(execute_mock, instance_mock):
    instance = SingularityInstance("/path/to/image")

    command = "uno dos tres catorce"
    response = instance.run(command, background=False)
    assert response is True
    assert instance._response_queue is None

    # TODO: figure out how to check this
    # expected_command = ["/bin/bash", "-c", command]
    # execute_mock.assert_called_once_with(command=expected_command)

    response = instance.run(command, background=True)
    assert response is None
    assert instance._response_queue.get_nowait() is True

    # TODO: same as above
    # execute_mock.assert_called_once_with(command=expected_command)

    with pytest.raises(ValueError):
        instance.run(command, background=True)


@patch("spython.main.Client.instance")
def test_singularity_instance_wait(instance_mock):
    instance = SingularityInstance("/path/to/image")

    with patch(
        "tritonclient.grpc.InferenceServerClient.is_server_live",
        return_value=True,
    ):
        # we don't have a background thread running yet,
        # so this should complain about that
        with pytest.raises(ValueError):
            instance.wait()

        # have the thread exit right away, but it shouldn't
        # matter because the first `is_server_live` will return
        # true and we'll be fine
        with patch("spython.main.Client.execute", return_value=True):
            instance.run("whatever", background=True)
            instance.wait()

    # this time the server will return False but
    # the process will be dead, so a value error
    # will get raised
    with patch(
        "tritonclient.grpc.InferenceServerClient.is_server_live",
        return_value=False,
    ):
        response = {"return_code": -99, "message": "oh no!"}
        instance._thread = instance._response_queue = None
        with patch("spython.main.Client.execute", return_value=response):
            instance.run("whatever", background=True)

            with pytest.raises(ValueError) as exc_info:
                instance.wait()
            assert "return code -99" in str(exc_info.value)

    # now make the thread take a second to return
    # and confirm that we can get a false before a true
    def wait_a_sec(*args, **kwargs):
        time.sleep(0.1)
        return True

    responses = iter([False, False, True])

    def is_server_live(obj):
        return next(responses)

    with patch(
        "tritonclient.grpc.InferenceServerClient.is_server_live",
        new=is_server_live,
    ):
        instance._thread = instance._response_queue = None
        with patch("spython.main.Client.execute", new=wait_a_sec):
            instance.run("whatever", background=True)
            instance.wait()
