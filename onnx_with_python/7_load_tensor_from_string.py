from onnx import load_tensor_from_string

with open("saved_tensor.pb", "rb") as f:
    serialized = f.read()

# 用load_tensor_from_string函数实现反序列化
proto = load_tensor_from_string(serialized)
print(type(proto))
