'''
前面的例子中，把线性回归的系数也作为模型的输入。这并不是非常合理。它们应当作为initializer（或常量）为模型的一部分，以符合ONNX的语义。
initializer就代表常量。
任何与输入同名的initializer都被视为默认值。如果未提供相应的输入，则使用该initializer来替代输入。
'''

import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# initializers
value = numpy.array([0.5, -0.6], dtype=numpy.float32)
A = numpy_helper.from_array(value, name='A')  # 调用numpy_helper.from_array创建initializer A。

value = numpy.array([0.4], dtype=numpy.float32)
C = numpy_helper.from_array(value, name='C')  # 调用numpy_helper.from_array创建initializer C。

# the part which does not change
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])  # 创建input X。
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])  # 创建output Y。
node1 = make_node('MatMul', ['X', 'A'], ['AX'])  # 创建node1，对应算子是’MatMul’，输入是[‘X’, ‘A’]，输出是[‘AX’]。
node2 = make_node('Add', ['AX', 'C'], ['Y'])  # 修建node2，对应算子是’Add’，输入是[‘AX’, ‘C’]，输出是[‘Y’]。
# 创建graph，对应节点是[node1, node2], graph名字为'lr'，输入是[X], 输出是[Y], initializer是[A, C]。
graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
# 创建onnx model。
onnx_model = make_model(graph)
check_model(onnx_model)  # 检查模型

print(onnx_model)
