# Hermes
## Deep Learning Inference-as-a-Service Deployment Utilties
`hermes` is a set of libraries for simplifying the deployment of deep learning applications via [Triton Inference Server (INSERT LINK)](https://github.com/triton-inference-server/server). Each library is installed and managed independently to keep deployments lightweight and minimize dependencies for use.
However, components are designed to play well together across libraries in order to minimize the overhead required to create new, exciting applications.

`hermes` is particularly aimed at streaming timeseries use cases, like those found in [gravitational wave](https://github.com/ML4GW/DeepClean) [physics](https://github.com/ML4GW/BBHNet). In particular, it includes helpful APIs for exposing input and output states on the server to minimize data I/O, as outlined in [INSERT NATURE PAPER]() and [INSERT FLEXSCIENCE PAPER]().

## The `hermes` libraries
### [`hermes.aeriel`](./hermes/hermes.aeriel)
#### Triton serving and client utilities
`aeriel` wraps Triton's `InferenceServerClient` class with neat functionality for inferring the names, shapes, and datatypes of the inputs required by complex ensembles of models with combinations of stateful and stateless inputs,
and exposing these inputs for asynchronous inference via numpy arrays.

The `aeriel.serve` submodule also includes a [Python context](INSERT LINK) for spinning up a local Triton inference service via [Singularity](), the preferred container runtime on the HPC clusters on which GW physics work typically takes place.

### [`hermes.cloudbreak`](./hermes/hermes.cloudbreak)
#### Cloud orchestration and deployment
`cloudbreak` contains utilities for orchestrating and deploying workloads on cloud-based resources via simple APIs in a cloud-agnostic manner (though only Google Cloud is supported as a backend [at the moment (INSERT LINK)](./issues/6)). This includes both Kubernetes clusters and swarms of VMs to perform parallel inference.

### [`hermes.quiver`](./hermes/hermes.quiver)
#### Model export and acceleration
`quiver` assists in exporting trained neural networks from both Torch and TensorFlow to either cloud or local [model repositories (INSERT LINK)](https://github.com/triton-inference-server/tree/main/docs), simplifying the creation of complex [model ensembles (INSERT LINK)](https://github.com/triton-inference-server/tree/main/docs) and server-side streaming innput and output states.

`quiver` also contains utilities for converting models from your framework of choice to NVIDIA's [TensorRT (INSERT LINK)]() inference library, which can sometimes help accelerate inference.

### [`hermes.stillwater`](./hermes/hermes/stillwater)
#### Asynchronous pipeline development
The `stillwater` submodule assists in building asychronous inference pipelines by leveraging Python multiprocessing and passing data from one process to the next, with wrappers around the client in `hermes.aeriel` to support truly asynchronous inference and response handling.


## Installation
Hermes is not [currently (INSERT ISSUE LINK)](./issues/) hosted on PyPI, so to install you'll need to clone this repo and add the submodule(s) you require via [Poetry (INSERT LINK)](https://python-poetry.org).

## Stability and Development
Hermes is still very much a work in progress, but the fastest path towards making it more robust is broader adoption! To that end, we warn users that they may experience bugs as they deploy Hermes to new and novel problems, and encourage them to file [issues](./issues) on this page and if they can, consider contributing a [PR (INSERT LINK)]() to fix whatever bug they stumbled upon!

Development of Hermes requires Poetry for managing and testing individual submodules. Moreover, it's highly encouraged to `poetry install` the root project, then run `poetry run pre-commit install --all` to install pre-commit hooks for style checking and static linting. For more information, see our [contribution guidelines](./CONTRIBUTING.md)
