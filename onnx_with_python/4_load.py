# 可以通过load函数导入磁盘上的ONNX模型

from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

# display
print(onnx_model)
