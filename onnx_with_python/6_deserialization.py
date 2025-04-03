from onnx import TensorProto
from onnx.numpy_helper import to_array

# 从磁盘读入序列化后的tensor。
with open("saved_tensor.pb", "rb") as f:
    serialized_tensor = f.read()
print(type(serialized_tensor))

onnx_tensor = TensorProto()
# 调用onnx_tensor.ParseFromString函数进行反序列化。
onnx_tensor.ParseFromString(serialized_tensor)
print(type(onnx_tensor))

# 调用onnx.numpy_helper.to_array函数将onnx tensor转换为numpy array。
numpy_tensor = to_array(onnx_tensor)
print(numpy_tensor)
