from dataset import MNIST
import torch 
from moe import MNIST_MoE
from config import *
from torch.utils.data import DataLoader
import time 

EPOCH=10
BATCH_SIZE=64 
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

dataset=MNIST() # 数据集
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,num_workers=10,persistent_workers=True)    # 数据加载器

model=MNIST_MoE(INPUT_SIZE,EXPERTS,TOP,EMB_SIZE).to(DEVICE) # 创建MNIST_MoE模型。
model.load_state_dict(torch.load('model.pth'))  # 加载权重

model.eval()    # 预测模式

start_time=time.time()
correct=0
for epoch in range(EPOCH):
    for img,label in dataloader:
        logits,_,_=model(img.to(DEVICE))  # 执行推理
        
        correct+=(logits.cpu().argmax(-1)==label).sum()
        
print('正确率:%.2f'%(correct/(len(dataset)*EPOCH)*100),'耗时:',time.time()-start_time,'s')