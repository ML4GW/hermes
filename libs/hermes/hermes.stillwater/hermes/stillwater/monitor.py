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
    "request"
]


def _get_re(prefix, model, version):
    return re.compile(
        "".join([
            prefix,
            f'(?P<gpu_id>{_uuid_pattern})",',
            f'model="{model}",',
            fr'version="{version}"\}} ',
            "(?P<value>[0-9.]+)",
        ])
    )


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
        if isinstance(ips, str):
            ips = [ips]

        self.res = {}
        self.trackers = {}
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
                versions = [i if i != -1 else 1 for i in versions]
            else:
                models = [model_name]
                versions = [model_version or 1]

            self.trackers[ip] = {}
            for model, version in zip(models, versions):
                prefix = _process_format.format("exec", "count")
                self.res[model] = {"count": _get_re(prefix, model, version)}
                self.trackers[ip][model] = {"count": {}}
                for process in _processes:
                    prefix = _process_format.format(process, "duration_us")
                    self.res[model][process] = _get_re(prefix, model, version)
                    self.trackers[ip][model][process] = {}

        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    def parse_for_ip(self, ip):
        response = requests.get(f"http://{ip}:8002/metrics")
        timestamp = time.time()

        response.raise_for_status()
        content = response.content.decode()

        lines = []
        for model, processes in self.res.items():
            tracker = self.trackers[ip][model]
            counts = {}
            for gpu_id, value in processes["count"].findall(content):
                value = int(float(value))
                try:
                    last = tracker["count"][gpu_id]
                except KeyError:
                    continue
                else:
                    diff = value - last
                    if diff > 0:
                        counts[gpu_id] = diff
                finally:
                    tracker["count"][gpu_id] = value

            durs = defaultdict(dict)
            for process in _processes:
                for gpu_id, value in processes[process].findall(content):
                    value = int(float(value))
                    try:
                        last = tracker[process][gpu_id]
                    except KeyError:
                        continue
                    else:
                        diff = value - last
                        durs[gpu_id][process] = diff
                    finally:
                        tracker[process][gpu_id] = value

            start = f"{timestamp},{ip},{model}"
            for gpu_id, processes in durs.items():
                try:
                    count = counts[gpu_id]
                except KeyError:
                    continue
                line = start + "," + gpu_id + "," + str(count)

                for process in _processes:
                    line += "," + str(processes[process])
                lines.append(line)
        return lines

    def write(self, content):
        with self.lock:
            self.f.write(content)

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
            self.f = open(self.filename, "w")
            header = "timestamp,ip,model,gpu_id,count,"
            header += ",".join(_processes)
            self.write(header)
            with futures.ThreadPoolExecutor(len(self.trackers)) as ex:
                fs = []
                for ip in self.trackers.keys():
                    fs.append(ex.submit(self.target, ip))

            futures.wait(fs, return_when=futures.FIRST_EXCEPTION)
        except Exception as e:
            self.cleanup(e)
            exitcode = 1
        finally:
            self.f.close()
            self.logger.debug("Target completed")
            listener.queue.close()
            listener.queue.join_thread()

            # exit the process with the indicated code
            sys.exit(exitcode)
