# Hermes
## Deep Learning Inference-as-a-Service Deployment Utilties
`hermes` is a set of libraries for simplifying the deployment of deep learning applications via [Triton Inference Server](https://github.com/triton-inference-server/server).

`hermes` is particularly aimed at streaming timeseries use cases, like those found in [gravitational](https://github.com/ML4GW/DeepClean) [wave](https://github.com/ML4GW/BBHNet) [physics](https://github.com/ml4gw/pe). In particular, it includes helpful APIs for exposing input and output states on the server to minimize data I/O, as outlined in [arXiv:2108.12430](https://arxiv.org/abs/2108.12430) and [doi.org/10.1145/3526058.3535454](https://dl.acm.org/doi/10.1145/3526058.3535454).

## The `hermes` modules
### [`hermes.aeriel`](./hermes/aeriel)
#### Triton serving and client utilities
The `aeriel.client` submodule wraps Triton's `InferenceServerClient` class with neat functionality for inferring the names, shapes, and datatypes of the inputs required by complex ensembles of models with combinations of stateful and stateless inputs,
and exposing these inputs for asynchronous inference via numpy arrays.

The `aeriel.serve` submodule also includes a Python context manager for spinning up a local Triton inference service via [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html), the preferred container runtime on the HPC clusters on which GW physics work typically takes place.

The `aeriel.monitor` submodule contains a `ServerMonitor` context manager for monitoring Triton server-side metrics such as model latency and throughput. This can be extremely useful for diagnosing and addressing bottlenecks in deployment configurations

### [`hermes.quiver`](./hermes/quiver)
#### Model export and acceleration
`quiver` assists in exporting trained neural networks from both Torch and TensorFlow to either cloud or local [model repositories](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md), simplifying the creation of complex [model ensembles](https://github.com/triton-inference-server/server/blob/main/docs/architecture.md#ensemble-models) and server-side streaming input and output states.

`quiver` also contains utilities for converting models from your framework of choice to NVIDIA's [TensorRT](https://developer.nvidia.com/tensorrt) inference library, which can sometimes help accelerate inference.

## Examples
### Local Triton Server with `hermes.aeriel.serve.serve` 

```python
from hermes.aeriel.serve import serve
from tritonclient import grpc as triton

with serve("/path/to/model/repository", "/path/to/container/image", wait=True):
    # wait ensures that the server comes online before we enter the context
    client = triton.InferenceServerClient("localhost:8001")
    assert client.is_server_live()

# exiting the context will spin down the server
try:
    client.is_server_live()
except triton.InferenceServerException:
    print("All done!")
```

You can even specify arbitrary GPUs to expose to Triton via the `CUDA_VISIBLE_DEVICES` environment variable:

```python
with serve(..., gpus=[0, 3, 5]):
    # do inference on 3 GPUs here
```

Note that since the mechanism for exposing these GPUs to Triton is by setting the `CUDA_VISIBLE_DEVICES` environment variable, the desired GPUs should be indexed by their _global_ indices, _not_ any indices mapped to by the current value of `CUDA_VISIBLE_DEVICES`.
For example, if `CUDA_VISIBLE_DEVICES=2,4,6,7` in my inference script's environment, setting `gpus=[0,2]` will expose the GPUs with global indices 0 and 2 to Triton, _not_ 2 and 5.

You can also choose to wait for the server at any time by using the `SingularityInstance` object returned by the `serve` context:

```python

with serve(..., wait=False) as instance:
    do_some_setup_while_we_wait_for_server()

    # now wait for the server before we begin the actual inference
    instance.wait()

    client = triton.InferenceServerClient("localhost:8001")
    assert client.is_server_live()
```

Consult the function's documentation for information about other configuration and logging options.
This function is not suitable for at-scale deployment, but is useful for running self-contained inference scripts for e.g. local model validation.

## Installation
Hermes is pip installable via `pip install ml4gw-hermes`. Hermes is also fully compatible with [uv](https://docs.astral.sh/uv/) for ease of use as a git submodule.


## Stability and Development
Hermes is still very much a work in progress, but the fastest path towards making it more robust is broader adoption! To that end, we warn users that they may experience bugs as they deploy Hermes to new and novel problems, and encourage them to file [issues](/../../issues) on this page and if they can, consider contributing a [PR](https://github.com/ML4GW/hermes/pulls) to fix whatever bug they stumbled upon!

Development of Hermes requires uv for managing and testing individual submodules. Moreover, it's highly encouraged to `uv sync` the root project, then run `uv run pre-commit install --all` to install pre-commit hooks for style checking and static linting. For more information, see our [contribution guidelines](./CONTRIBUTING.md)
