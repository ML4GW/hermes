import re
import sys
import threading
import time
from collections import defaultdict
from concurrent import futures
from typing import Optional, Sequence, Union

import requests
import tritonclient.grpc as triton

from hermes.stillwater.logging import listener
from hermes.stillwater.process import PipelineProcess

_uuid_pattern = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_process_format = r'(?<=nv_inference_{}_{}\{{gpu_uuid="GPU-)'
_processes = [
    "queue",
    "compute_input",
    "compute_infer",
    "compute_output",
    "exec",
]
_prefixes = {
    p: {
        "duration": _process_format.format(p, "duration_us"),
        "count": _process_format.format(p, "count"),
    }
    for p in _processes
}


class ServerMonitor(PipelineProcess):
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

        self.res = {}
        self.tracker = {}
        for ip in ips:
            client = triton.InferenceServerClient(f"{ip}:8001")
            config = client.get_model_config(model_name).config

            if config.platform == "ensemble":
                mvs = list(
                    set(
                        [
                            (step.model_name, step.model_version)
                            for step in config.ensemble_scheduling.step
                        ]
                    )
                )
                models, versions = zip(*mvs)
            else:
                models = [model_name]
                versions = [model_version]

            self.trackers[ip] = {}
            for model, version in zip(models, versions):
                self.res[model] = {}
                self.trackers[ip][model] = {}
                for process, prefixes in _prefixes.items():
                    self.res[model][process] = {}
                    self.trackers[ip][model][process] = {}
                    for metric, prefix in prefixes.items():
                        reg = "".join(
                            [
                                prefix,
                                f'(?P<gpu_id>{_uuid_pattern})",',
                                f'model="{model}",'
                                fr'version="{version}"\}} ',
                                "(?P<value>[0-9.]+)",
                            ]
                        )
                        self.res[model][process][metric] = re.compile(reg)
                        self.trackers[ip][model][process][metric] = {}

        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    def parse_for_ip(self, ip):
        response = requests.get(f"http://{ip}:8002/metrics")
        timestamp = time.time()

        response.raise_for_status()
        content = response.content.decode()

        lines = []
        for model, processes in self.res.items():
            for process, metrics in processes.items():
                start = f"{timestamp},{ip},{model},{process}"
                diffs = defaultdict(defaultdict(dict))
                for metric, regex in metrics.items():
                    tracker = self.trackers[ip][model][process][metric]
                    for gpu_id, value in regex.findall(content):
                        value = int(value)
                        try:
                            last = tracker[gpu_id]
                        except KeyError:
                            continue
                        else:
                            diff = value - last
                        finally:
                            tracker[gpu_id] = value

                        diffs[gpu_id][metric] = diff

                for gpu_id, metrics in diffs.items():
                    if metrics["count"] == 0:
                        continue

                    line = start + "," + gpu_id
                    for metric in ["count", "duration"]:
                        line += "," + str(metrics[metric])
                    lines.append(line)
        return lines

    def write(self, content):
        with self.lock.acquire():
            with open(self.filename, "a") as f:
                f.write(content)

    def target(self, ip):
        try:
            while not self.stopped:
                lines = self.parse_for_ip(ip)
                self.write("\n" + "\n".join(lines))
        except Exception as e:
            self.logger.error(f"Encountered error in parsing ip {ip}:\n {e}")
            self.stop()
            raise

    def run(self):
        self.logger = listener.add_process(self)
        exitcode = 0
        try:
            self.write("timestamp,ip,model,process,count,duration_us")
            with futures.ThreadPoolExecutor(len(self.tracker)) as ex:
                fs = []
                for ip in self.trackers.keys():
                    fs.append(ex.submit(self.target, ip))

            futures.wait(fs, return_when=futures.FIRST_EXCEPTION)
        except Exception as e:
            self.cleanup(e)
            exitcode = 1
        finally:
            self.logger.debug("Target completed")
            listener.queue.close()
            listener.queue.join_thread()

            # exit the process with the indicated code
            sys.exit(exitcode)
