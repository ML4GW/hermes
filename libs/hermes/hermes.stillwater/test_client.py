import logging
import sys
import time

import numpy as np

from hermes.stillwater.client import InferenceClient
from hermes.stillwater.utils import Package

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    with InferenceClient(
        "34.145.125.159:8001",
        model_name="ensemble",
        model_version=1,
        name="client",
        batch_size=8,
    ) as client:
        i = 0
        for i in range(1000):
            t0 = time.time()

            packages = {}
            for _, channel_map in client.states:
                for name, shape in channel_map.items():
                    x = np.random.randn(*shape).astype("float32")
                    package = Package(
                        x=x,
                        t0=t0,
                        sequence_id=1001,
                        sequence_start=i == 0,
                        sequence_end=i == 999,
                    )
                    packages[name] = package
            for input in client.inputs:
                x = np.random.randn(*input.shape()).astype("float32")
                packages[input.name()] = Package(x=x, t0=t0)

            client.in_q.put(packages)
            time.sleep(0.001)
        client.in_q.put(StopIteration())

        for n, j in enumerate(client):
            if (n + 1) % 50 == 0:
                logging.debug("Completed {}/{}".format(n + 1, 1000))
        logging.info(f"Returned {n+1} requests")
        logging.info(j)

