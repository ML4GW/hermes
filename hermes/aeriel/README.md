# `hermes.aeriel`
## Triton serving and client utilities
`hermes.aeriel` exposes simplified APIs for running a Triton inference service locally for small scale testing and deployment, and for making asynchronous streaming inference requests to _any_ Triton instance (local or otherwise).

### `serve` submodule
The most important function here is the `hermes.aeriel.serve.serve` context, which will use Singularity's Python APIs to spin up a local Singularity container instance running Triton:

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
