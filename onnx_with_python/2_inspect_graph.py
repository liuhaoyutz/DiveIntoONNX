from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

'''
调用make_tensor_value_info创建input。
第一个参数name代表tensor的名字，如”X”, “A”, “B”。
第二个参数elem_type是tensor data type，如TensorProto.FLOAT，TensorProto.FLOAT.UINT8。
第三个参数shape是tensor的形状，[None, None]代表未定义形状，可以是任意形状。
'''
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

# 调用make_tensor_value_info创建output。
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

'''
调用make_node函数创建node
第一个参数op_type指定node对应的算子类型，例如”MatMul”，”Add”。
第二个参数inputs是input names列表，例如['X', 'A']，表示node1的input是'X'和'A'。
第三个参数outputs是output names列表，例如 ['XA']，表示node1的output是’XA’。
'''
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

'''
调用make_graph函数创建graph
第一个参数nodes是node列表，例如[node1, node2]。
第二个参数name是graph的名字，例如'lr'。
第三个参数inputs是input列表，例如[X, A, B]。
第四个参数outputs是output列表，例如[Y]。
'''
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])

# 调用make_model函数创建模型
onnx_model = make_model(graph)
# 检查模型
check_model(onnx_model)

# the list of inputs
print('** inputs **')
print(onnx_model.graph.input)  # 打印onnx_model.graph.input

# 调用shape2tuple，以更容易理解的方式打印onnx_model.graph.input
# in a more nicely format
print('** inputs **')
for obj in onnx_model.graph.input:
    print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))

# the list of outputs
print('** outputs **')
print(onnx_model.graph.output)  # 打印onnx_model.graph.output

# 调用shape2tuple，以更容易理解的方式打印onnx_model.graph.output
# in a more nicely format
print('** outputs **')
for obj in onnx_model.graph.output:
    print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))

# the list of nodes
print('** nodes **')
print(onnx_model.graph.node)  # 打印onnx_model.graph.node

# 以更容易理解的方式打印onnx_model.graph.node
# in a more nicely format
print('** nodes **')
for node in onnx_model.graph.node:
    print("name=%r type=%r input=%r output=%r" % (
        node.name, node.op_type, node.input, node.output))
