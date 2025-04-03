import numpy
from onnx.numpy_helper import from_array

numpy_tensor = numpy.array([0, 1, 4, 5, 3], dtype=numpy.float32)
print(type(numpy_tensor))

# 将numpy tensor转换为onnx tensor
onnx_tensor = from_array(numpy_tensor)
print(type(onnx_tensor))

# 调用onnx_tensor.SerializeToString函数将onnx_tensor序列化。
serialized_tensor = onnx_tensor.SerializeToString()
print(type(serialized_tensor))

# 将序列化后的serialized_tensor保存到saved_tensor.pb文件中。
with open("saved_tensor.pb", "wb") as f:
    f.write(serialized_tensor)
