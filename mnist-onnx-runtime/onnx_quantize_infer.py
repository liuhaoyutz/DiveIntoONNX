from torch import nn 
import torch 
import onnx 
import netron
import onnxruntime
from moe import MNIST_MoE
from config import * 
from dataset import MNIST
from torch.utils.data import DataLoader
import time 
from onnxruntime.quantization import QuantType, quantize_dynamic
import os 

EPOCH=10
BATCH_SIZE=64 

dataset=MNIST() # 数据集
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,num_workers=10,persistent_workers=True)    # 数据加载器

model=MNIST_MoE(INPUT_SIZE,EXPERTS,TOP,EMB_SIZE) # 创建MNIST_MoE模型
model.load_state_dict(torch.load('model.pth'))  # 导入权重

# torch.jit.script是PyTorch提供的一种工具，用于将包含动态控制流（如if语句、for循环等）的模型转换为TorchScript格式的静态计算图。
# 这种转换可以让模型脱离Python环境运行，比如在C++中部署。
# model=torch.jit.script(model) # 带控制流的静态图导出

model.eval()    # 预测模式

# 将模型导出为onnx格式。torch.onnx.export的工作是跟踪计算图。
# 这一步操作会运行参数model指定的模型，把第二个参数指定的input输入模型运行，得到计算图。因为PyTorch是动态图，只有运行时才能得到计算图。
torch.onnx.export(model,torch.rand((BATCH_SIZE,1,28,28)),f='model.onnx')

# 检查onnx导出正确
onnx_model=onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)

# 量化
os.system('python -m onnxruntime.quantization.preprocess --input model.onnx --output  model-shape.onnx --auto_merge')
quantize_dynamic('model-shape.onnx','model-quantize.onnx')
#quantize_dynamic('model-shape.onnx','model-quantize-QUInt4.onnx', weight_type=QuantType.QUInt4)

# 初始化onnxruntime session，使用CUDA或CPU后端
sess=onnxruntime.InferenceSession('model-quantize.onnx',providers=['CUDAExecutionProvider','CPUExecutionProvider'])

start_time=time.time()

correct=0
for epoch in range(EPOCH):
    for img,label in dataloader:
        batch_size=img.size(0)
        if img.size(0)!=BATCH_SIZE: # onnx输入尺寸固定，最后1个batch要补齐
            fills=torch.zeros(BATCH_SIZE-img.size(0),1,28,28)
            img=torch.concat((img,fills),dim=0)
        # 通过onnxruntime session进行推理。
        outputs=sess.run(output_names=None,input_feed={sess.get_inputs()[0].name:img.numpy()})  # 输入&输出
        logits=outputs[0][:batch_size]

        correct+=(logits.argmax(-1)==label.numpy()).sum()

print('正确率:%.2f'%(correct/(len(dataset)*EPOCH)*100),'耗时:',time.time()-start_time,'s')