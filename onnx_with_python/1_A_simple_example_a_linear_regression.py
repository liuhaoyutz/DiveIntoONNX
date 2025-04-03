# imports

from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# inputs
'''
调用make_tensor_value_info创建input。
第一个参数name代表tensor的名字，如”X”, “A”, “B”。
第二个参数elem_type是tensor data type，如TensorProto.FLOAT，TensorProto.FLOAT.UINT8。
第三个参数shape是tensor的形状，[None, None]代表未定义形状，可以是任意形状。
'''
# 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

# 调用make_tensor_value_info创建output。
# outputs, the shape is left undefined
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# nodes
'''
调用make_node函数创建node
第一个参数op_type指定node对应的算子类型，例如”MatMul”，”Add”。
第二个参数inputs是input names列表，例如['X', 'A']，表示node1的input是'X'和'A'。
第三个参数outputs是output names列表，例如 ['XA']，表示node1的output是’XA’。
'''
# It creates a node defined by the operator type MatMul,
# 'X', 'A' are the inputs of the node, 'XA' the output.
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

# from nodes to graph
# the graph is built from the list of nodes, the list of inputs,
# the list of outputs and a name.
'''
调用make_graph函数创建graph
第一个参数nodes是node列表，例如[node1, node2]。
第二个参数name是graph的名字，例如'lr'。
第三个参数inputs是input列表，例如[X, A, B]。
第四个参数outputs是output列表，例如[Y]。
'''
graph = make_graph([node1, node2],  # nodes
                    'lr',  # a name
                    [X, A, B],  # inputs
                    [Y])  # outputs

# onnx graph
# there is no metadata in this case.
# 调用make_model函数创建模型
onnx_model = make_model(graph)

# Let's check the model is consistent,
# this function is described in section
# Checker and Shape Inference.
check_model(onnx_model)  # 检查模型

# the work is done, let's display it...
print(onnx_model)  # 打印onnx_model的内容
