import onnx
from onnx import shape_inference

model = onnx.load('/Downloads/onnx_models/resnet101-v1-7.onnx')
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "/Downloads/onnx_models/resnet101-v1-7.onnx")
