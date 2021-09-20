from enum import Enum


class Platform(Enum):
    ONNX = "onnxruntime_onnx"
    SAVEDMODEL = "tensorflow_savedmodel"
    TENSORRT = "tensorrt_plan"
    ENSEMBLE = "ensemble"


conventions = {
    Platform.ONNX: "model.onnx",
    Platform.SAVEDMODEL: "model.savedmodel",
    Platform.TENSORRT: "model.plan",
    Platform.ENSEMBLE: "model.empty",
}
