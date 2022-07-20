from pathlib import Path
from unittest.mock import MagicMock, patch

from google.protobuf import text_format
from tritonclient.grpc.model_config_pb2 import ModelConfig

from hermes.stillwater.monitor import ServerMonitor, _processes

data_path = Path(__file__).resolve().parent / "monitor_data"
metadata_mock = MagicMock()
metadata_mock.versions = [1]


def get_config_patch(obj, model_name):
    config_path = data_path / f"{model_name}_config.pbtxt"
    config = ModelConfig()
    with open(config_path, "r") as f:
        text_format.Merge(f.read(), config)

    mock = MagicMock()
    mock.config = config
    return mock


def get_response(i: int):
    response_path = data_path / f"t{i}.txt"
    with open(response_path, "r") as f:
        content = f.read()

    response = MagicMock()
    response.data.decode = MagicMock(return_value=content)
    return response


@patch(
    "tritonclient.grpc.InferenceServerClient.get_model_config",
    new=get_config_patch,
)
def test_server_monitor_on_standalone_model():
    monitor = ServerMonitor(
        "test-model",
        ips="localhost",
        filename="test.csv",
        model_version=1,  # TODO: test -1 behavior
        name="monitor",
    )

    http = MagicMock()
    http.request = MagicMock(return_value=get_response(0))
    tracker = {"test-model": {}}

    result = monitor.parse_for_ip("localhost", http, tracker)
    assert len(result) == 0
    assert len(tracker["test-model"]) == (len(_processes) + 1)
    for process in _processes + ["count"]:
        assert tracker["test-model"][process] == 0

    http.request = MagicMock(return_value=get_response(1))
    result = monitor.parse_for_ip("localhost", http, tracker)
    assert len(result) == 1

    result = result[0].split(",")
    assert len(result) == 9
    assert result[1] == "localhost"
    assert result[2] == "test-model"
    assert int(result[3]) == tracker["test-model"]["count"] == 10
    assert int(result[-1]) == tracker["test-model"]["request"] == 7038


@patch(
    "tritonclient.grpc.InferenceServerClient.get_model_config",
    new=get_config_patch,
)
@patch(
    "tritonclient.grpc.InferenceServerClient.get_model_metadata",
    return_value=metadata_mock,
)
def test_server_monitor_on_ensemble_model(mock1):
    monitor = ServerMonitor(
        "ensemble",
        ips="localhost",
        filename="test.csv",
        model_version=1,  # TODO: test -1 behavior
        name="monitor",
    )

    assert set(monitor.models) == set(["stream", "test-model"])
    assert monitor.versions == [1, 1]

    http = MagicMock()
    http.request = MagicMock(return_value=get_response(0))
    tracker = {i: {} for i in monitor.models}

    result = monitor.parse_for_ip("localhost", http, tracker)
    assert len(result) == 0
    for model in monitor.models:
        assert len(tracker[model]) == (len(_processes) + 1)
        for process in _processes + ["count"]:
            assert tracker[model][process] == 0

    http.request = MagicMock(return_value=get_response(1))
    result = monitor.parse_for_ip("localhost", http, tracker)
    assert len(result) == 2

    latencies = {"stream": 75590, "test-model": 7038}
    for row, model, latency in zip(result, monitor.models, latencies):
        row = row.split(",")
        assert len(row) == 9
        assert row[1] == "localhost"
        assert row[2] == model
        assert int(row[3]) == tracker[model]["count"] == 10
        assert int(row[-1]) == tracker[model]["request"] == latencies[model]

    http.request = MagicMock(return_value=get_response(2))
    result = monitor.parse_for_ip("localhost", http, tracker)
    assert len(result) == 1

    row = result[0].split(",")
    assert row[2] == "test-model"
    assert int(row[3]) == 5
    assert tracker["test-model"]["count"] == 15
    assert int(row[-1]) == 1516
    assert tracker["test-model"]["request"] == 8554

    assert tracker["stream"]["count"] == 10
    assert tracker["stream"]["request"] == 75590
