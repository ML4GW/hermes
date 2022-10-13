import re
import sys
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Dict, Iterable, List, Union

import tritonclient.grpc as triton
import urllib3

from hermes.stillwater.logging import listener
from hermes.stillwater.process import PipelineProcess

if TYPE_CHECKING:
    from io import TextIOWrapper

_processes = [
    "queue",
    "compute_input",
    "compute_infer",
    "compute_output",
    "request",
]


def _get_re(process: str, metric: str, model: str, version: int):
    """Build a regex for find the GPU id and value in metrics rows"""

    prefix = f"nv_inference_{process}_{metric}"
    identifier = rf'\{{model="{model}",version="{version}"\}}'
    return re.compile(f"(?m)(?<={prefix}{identifier} )[0-9.]+$")


class ServerMonitor(PipelineProcess):
    """Process for monitoring server-side metrics and writing them to a file.

    Queries the Triton metrics endpoint at potentially
    multiple IP addresses and records server-side latency
    and throughput statistics at intervals for analyzing
    server performance. Standard Triton conventions for
    port mapping are expected: gRPC requests to port 8001
    and metrics requests to port 8002. Data is recorded to
    a csv file with columns for (in order)
        - the time at which the metrics are queried
        - the ip address metrics were queried from
        - the model to which this row of metrics apply
        - the number of inference executions of that model
            which occured since the last query
        - the time requests spent queueing for inference on
            that model since the last query in microseconds
        - the time spent populating the input tensor for
            that model since the last query in microseconds
        - the time spent executing the model inference for
            that model since the last query in microseconds
        - the time spent populating the model output tensor
            for that model since the last query in microseconds
        - the total time spent executing all inference
            processes for that model since the last query
            in microseconds

    Ensemble models are implicitly divided into their
    constituent models for analysis.

    Args:
        model_name:
            The name of the model whose inference metrics to record
        ips:
            The IP addresses of Triton servers from which
            to query inference metrics. If a string, only
            that IP address will have its metrics queried.
        filename:
            The name of the file to which to write results in
            CSV format
        model_version:
            The version of the model whose metrics to query. If
            the specified model is an ensemble model, this will
            be ignored. Otherwise, if left as -1, the latest
            available version of the model will be used.
        max_request_rate:
            The maximum rate at which to request metrics from
            each inference service. The actual rate may be lower
            due to download and parsing latencies, as well as
            threading overhead.
        name:
            Name to give to this process
        **kwargs:
            Additional keyword arguments to pass to the
            `PipelineProcess` parent class
    """

    def __init__(
        self,
        model_name: str,
        ips: Union[str, Iterable[str]],
        filename: str,
        model_version: int = -1,
        max_request_rate: float = 10,
        **kwargs,
    ) -> None:
        self.filename = filename
        if isinstance(ips, str):
            ips = [ips]

        # infer the names and versions of the models
        # we want to be requesting from each deployment
        self.ips = list(ips)
        self.models = self.version = None
        for ip in self.ips:
            # get the config of the model on each IP to
            # figure out which models we need to monitor
            # TODO: make port configurable
            client = triton.InferenceServerClient(f"{ip}:8001")
            config = client.get_model_config(model_name).config

            if config.platform == "ensemble":
                # if the specified model is an ensemble model,
                # then use the config to find the names and
                # versions of the models that we should monitor
                steps = config.ensemble_scheduling.step
                models = [i.model_name for i in steps]
                versions = [i.model_version for i in steps]

                # make sure a model doesn't appear more than once
                # TODO: is this necessary? Or was there another
                # reason for adding this that I'm forgetting?
                uniques = set(zip(models, versions))

                # now go through and reaggregate our model
                # names and versions, checking for models
                # that indicate that they would like to use
                # the latest version and inferring that version
                models, versions = [], []
                for model, version in list(uniques):
                    models.append(model)

                    if version == -1:
                        metadata = client.get_model_metadata(model)
                        version = max(map(int, metadata.versions))
                    versions.append(version)

                versions = [1 if i == -1 else i for i in versions]
            else:
                # otherwise just use the provided name and version explicitly,
                # inferring the latest version if -1 was passed
                models = [model_name]
                if model_version == -1:
                    metadata = client.get_model_metadata(model)
                    model_version = max(map(int, metadata.versions))
                versions = [model_version]

            # if we haven't recorded any models or versions for any
            # ip addresses yet, record them now. Otherwise, ensure
            # that the names and versions of models are the same
            # on all deployments. TODO: should this be relaxed?
            # Presumably the use case in mind here is a fleet of
            # services all reading from the same centralized
            # model repository, so there should be no reason
            # for them to be different
            if self.models is None:
                self.models = models
                self.versions = versions
            else:
                if set(self.models) != set(models):
                    raise ValueError(
                        "Model names {} on service at address {} "
                        "don't match up with inferred names {}".format(
                            ",".join(models), ip, ",".join(self.models)
                        )
                    )
                elif set(self.versions) != set(versions):
                    raise ValueError(
                        "Model versions {} on service at address {} "
                        "don't match up with inferred names {}".format(
                            ",".join(versions), ip, ",".join(self.versions)
                        )
                    )

        super().__init__(**kwargs)
        self.filename = filename
        self.max_request_rate = max_request_rate

    def parse_for_ip(
        self, ip: str, http: urllib3.PoolManager, tracker: Dict[str, int]
    ) -> List[str]:
        # request some metrics data from the given IP
        # and record the time at which we get the response
        # TODO: make port configurable
        response = http.request("GET", f"http://{ip}:8002/metrics")
        timestamp = time.time()
        content = response.data.decode()

        lines = []
        for model, version in zip(self.models, self.versions):
            try:
                model_tracker = tracker[model]
            except KeyError:
                raise ValueError(
                    "Tracker for models {} can't track model {}".format(
                        ",".join(list(tracker)), model
                    )
                )

            # for each model, first find out how many times
            # that model was executed on each GPU
            count_re = _get_re("exec", "count", model, version)
            count = count_re.search(content)
            if count is None:
                # sometimes Triton won't be able to collect metrics,
                # so raise an error if there's no data available
                raise ValueError(
                    "Couldn't find count for model {} version {} "
                    "on server at IP address {}. Metric service "
                    "response was:\n{}".format(model, version, ip, content)
                )

            count = int(float(count.group(0)))
            try:
                last = model_tracker["count"]
            except KeyError:
                # we haven't recorded executions for this
                # model yet, so there's no diff to record.
                # Leave this as `None` though so that we
                # make sure to go through and record the
                # existing durations for each process
                diff = None
            else:
                # compute the number of inferences executed
                # since the last request we made
                diff = count - last

            model_tracker["count"] = count
            if diff == 0:
                # don't bother if no new inferences happened
                continue
            elif diff is not None:
                # if we have new inferences to record, create
                # a new line to add to our running dataframe
                line = f"{timestamp},{ip},{model},{diff}"
            else:
                line = None

            # now collect the amount of time spent on each subprocess
            for process in _processes:
                dur_re = _get_re(process, "duration_us", model, version)
                # TODO: catch a miss here
                duration = int(float(dur_re.search(content).group(0)))

                try:
                    last = model_tracker[process]
                except KeyError:
                    # we don't have an entry for this process yet, so
                    # record one in the `finally` clause but then move on
                    continue
                else:
                    # add the duration for this process to our dataframe entry
                    diff = duration - last
                    line += "," + str(diff)
                finally:
                    model_tracker[process] = duration

            if line is not None:
                lines.append(line)
        return lines

    def target(
        self,
        ip: str,
        http: urllib3.PoolManager,
        f: "TextIOWrapper",
        lock: threading.Lock,
    ) -> None:
        """Thread target for iteratively collecting metrics from a service"""

        # since Triton metrics are cumulative, use a dict
        # to keep track of the values at each iteration so
        # that we can record the diffs
        tracker = {model: {} for model in self.models}
        try:
            while not self.stopped:
                lines = self.parse_for_ip(ip, http, tracker)
                if lines:
                    with lock:
                        f.write("\n" + "\n".join(lines))
                        f.flush()
                time.sleep(1 / self.max_request_rate)
        except Exception as e:
            self.logger.error(f"Encountered error in parsing ip {ip}:\n {e}")
            self.stop()
            raise

    def run(self) -> None:
        self.logger = listener.add_process(self)
        exitcode = 0
        fs = []
        try:
            # open a file for writing and add a csv header to it
            f = open(self.filename, "w")
            header = "timestamp,ip,model,count," + ",".join(_processes)
            f.write(header)

            # for each IP address, start up a thread that
            # pull the metrics for its own IP and writes
            # to the file in a thread-safe manner
            http = urllib3.PoolManager()
            lock = threading.Lock()
            with ThreadPoolExecutor(len(self.ips)) as ex:
                args = (http, f, lock)
                fs = [ex.submit(self.target, i, *args) for i in self.ips]

            # wait until all the threads are done
            # (i.e. self.stopped == True, or an exception
            # occurs on any of the threads)
            result = wait(fs, return_when=FIRST_EXCEPTION)

            # check to see if any of them stopped due to an exception
            list(result.done)[0].exception()
        except Exception as e:
            self.cleanup(e)
            exitcode = 1
        finally:
            # stop the threads and wait for them all to finish
            self.stop()
            wait(fs)

            # now close the file now that nothing else is
            # going to write to it
            f.close()

            # close the connection to the logger
            self.logger.debug("Target completed")
            listener.queue.close()
            listener.queue.join_thread()

            # exit the process with the indicated code
            sys.exit(exitcode)
