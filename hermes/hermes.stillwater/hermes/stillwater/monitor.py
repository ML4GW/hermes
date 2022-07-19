import logging
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent import futures
from typing import List, Optional, Sequence, Union

import requests
import tritonclient.grpc as triton

from hermes.stillwater.logging import listener
from hermes.stillwater.process import PipelineProcess

_uuid_pattern = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_process_format = r'(?<=nv_inference_{}_{}\{{)'  # gpu_uuid="GPU-)'
_processes = [
    "queue",
    "compute_input",
    "compute_infer",
    "compute_output",
    "request",
]


def _get_re(prefix: str, model: str, version: int):
    """Build a regex for find the GPU id and value in metrics rows"""

    return re.compile(
        "".join(
            [
                prefix,
                # f'(?P<gpu_id>{_uuid_pattern})",',
                f'model="{model}",',
                rf'version="{version}"\}} ',
                "(?P<value>[0-9.]+)",
            ]
        )
    )


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
        - the ID of the GPU to which this row of metrics apply
        - the number of inference executions of that model
            on that GPU which occured since the last query
        - the time requests spent queueing for inference on
            that model on that GPU since the last query in
            microseconds
        - the time spent populating the input tensor for
            that model on that GPU since the last query in
            microseconds
        - the time spent executing the model inference for
            that model on that GPU since the last query in
            microseconds
        - the time spent populating the model output tensor
            for that model on that GPU since the last query
            in microseconds
        - the total time spent executing all inference
            processes for that model on that GPU since the
            lsat query in microseconds

    Ensemble models are implicitly divided into their
    constituent models for analysis. It's worth noting for
    now that steps in an ensemble model that neglect to
    specify a version will use version 1 by default (
    **NOT** the latest version). This is a known limitation
    that is in the process of being fixed.

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
            be ignored. Otherwise, if left as `None`, the default
            version will be 1 (**NOT** the latest version).
    """

    def __init__(
        self,
        model_name: str,
        ips: Union[str, Sequence[str]],
        filename: str,
        model_version: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        self.filename = filename
        if isinstance(ips, str):
            ips = [ips]

        self.res = {}
        self.trackers = {}
        for ip in ips:
            # get the config of the model on each IP to
            # figure out which models we need to monitor
            # TODO: is there a way to get the config without
            # resorting to the client? This can cause problems
            # if running on the same node as the client if the
            # stream has already been established
            client = triton.InferenceServerClient(f"{ip}:8001")
            config = client.get_model_config(model_name).config

            if config.platform == "ensemble":
                # if the specified model is an ensemble model,
                # then use the config to find the names and
                # versions of the models that we should monitor
                models, versions = zip(
                    *list(
                        set(
                            [
                                (step.model_name, step.model_version)
                                for step in config.ensemble_scheduling.step
                            ]
                        )
                    )
                )

                # use default value of 1 if config step doesn't
                # specify a version. TODO: this is _NOT_ good
                # general behavior, since the -1 indicates that
                # we're supposed to use the latest version
                versions = [i if i != -1 else 1 for i in versions]
            else:
                # otherwise just use the provided name and
                # version explicitly
                models = [model_name]
                versions = [model_version or 1]

            # since Triton metrics are cumulative, for each
            # IP address, keep track of the most recent value
            # for each process for each model on each GPU
            self.trackers[ip] = {}
            for model, version in zip(models, versions):
                # get the regex for reading the total number
                # of inferences performed
                prefix = _process_format.format("exec", "count")
                self.res[model] = {"count": _get_re(prefix, model, version)}
                self.trackers[ip][model] = {"count": {}}

                # for each subprocess, build an re that can collect
                # the total amount of time spent executing that
                # particular process in microseconds
                for process in _processes:
                    prefix = _process_format.format(process, "duration_us")
                    self.res[model][process] = _get_re(prefix, model, version)
                    self.trackers[ip][model][process] = {}

        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()
        self._f = None

    def parse_for_ip(self, ip: str) -> List[str]:
        # request some metrics data from the given IP
        # and record the time at which we get the response
        response = requests.get(f"http://{ip}:8002/metrics")
        timestamp = time.time()

        # check for HTTP errors then decode the response
        response.raise_for_status()
        content = response.content.decode()
 
        lines = []
        for model, processes in self.res.items():
            tracker = self.trackers[ip][model]

            # for each model, first find out how many times
            # that model was executed on each GPU
            counts = {}
            matches = processes["count"].findall(content)
            gpu_id = "1"
            # for gpu_id, value in matches:
            for value in matches:
                value = int(float(value))
                try:
                    last = tracker["count"][gpu_id]
                except KeyError:
                    # we haven't recorded executions for this
                    # model on this GPU yet, so there's no
                    # diff to record
                    continue
                else:
                    # compute the number of inferences executed
                    # since the last request we made. Only
                    # record it's nonzero
                    diff = value - last
                    if diff > 0:
                        counts[gpu_id] = diff
                finally:
                    # no matter what happens, update our count
                    # tracker for this model and GPU
                    tracker["count"][gpu_id] = value

            # now collect the amount of time spent
            # on each subprocess for each GPU
            durs = defaultdict(dict)
            for process in _processes:
                matches = processes[process].findall(content)
                # for gpu_id, value in matches:
                for value in matches:
                    value = int(float(value))
                    try:
                        last = tracker[process][gpu_id]
                    except KeyError:
                        # no data yet for this GPU, so no diff to record
                        continue
                    else:
                        # calculate the total number of microseconds spent
                        # executing this process since the last request
                        diff = value - last
                        durs[gpu_id][process] = diff
                    finally:
                        # no matter what happens, update the tracker for
                        # this model/process/GPU combo
                        tracker[process][gpu_id] = value

            # now for each update we need to record for each
            # GPU, create a new line to add to our dataframe
            start = f"{timestamp},{ip},{model}"
            for gpu_id, processes in durs.items():
                try:
                    count = counts[gpu_id]
                except KeyError:
                    # either we didn't have a value to compute
                    # a diff from, or the diff was 0, so nothing
                    # to record here
                    continue

                # complete this row of the dataframe with
                # GPU ID, count, and duration data
                line = start + "," + gpu_id + "," + str(count)
                for process in _processes:
                    line += "," + str(processes[process])
                lines.append(line)
        return lines

    def write(self, content: str) -> None:
        """Thread-safe write of data from a particular IP"""

        if self._f is None:
            raise ValueError("Must have opened internal file to write")

        with self.lock:
            self._f.write(content)

    def target(self, ip):
        """Thread target for iteratively collecting metrics from a service"""

        try:
            while not self.stopped:
                lines = self.parse_for_ip(ip)
                if lines:
                    self.write("\n" + "\n".join(lines))
                time.sleep(0.1)
        except Exception as e:
            if self.stopped:
                return
            self.logger.error(f"Encountered error in parsing ip {ip}:\n {e}")
            self.stop()
            raise

    def run(self):
        self.logger = listener.add_process(self)
        exitcode = 0
        try:
            # open a file for writing and add a csv header to it
            self._f = open(self.filename, "w")
            header = "timestamp,ip,model,gpu_id,count,"
            header += ",".join(_processes)
            self.write(header)

            # for each IP address, start up a thread that
            # pull the metrics for its own IP and writes
            # to the file in a thread-safe manner
            with futures.ThreadPoolExecutor(len(self.trackers)) as ex:
                fs = []
                for ip in self.trackers.keys():
                    fs.append(ex.submit(self.target, ip))

            # wait until all the threads are done
            # (i.e. self.stopped == True, or an exception
            # occurs on any of the threads)
            futures.wait(fs, return_when=futures.FIRST_EXCEPTION)
        except Exception as e:
            self.cleanup(e)
            exitcode = 1
        finally:
            self.stop()
            self._f.close()
            self._f = None

            self.logger.debug("Target completed")
            listener.queue.close()
            listener.queue.join_thread()

            # exit the process with the indicated code
            sys.exit(exitcode)
