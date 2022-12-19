ARG TRITON_TAG
FROM nvcr.io/nvidia/tritonserver:${TRITON_TAG}-py3
ARG TENSORRT_VERSION
LABEL tensorrt_version=${TENSORRT_VERSION}
