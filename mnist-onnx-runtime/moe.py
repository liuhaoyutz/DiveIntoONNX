'''
对于待计算的输入向量emb:
1. 准备n个expert子网络
2. 准备gateway linear层，输入emb可以输出n个概率值
3. 选择n个概率中最高的top个位置，作为被选中的expert子网络
4. emb分别被top个expert子网络运算，输出的emb分别与各自的(gateway输出的)概率值相乘，并pointwise相加，得到最终的输出向量emb
'''

from torch import nn 
from torch import softmax
import torch 

'''
一个Expert类就是一个专家，它只是对输入x做2次线性变换（中间有做一次非线性变换的ReLU激活）。
线性变换也没有先拉宽再收窄，输入输出维度都是emb_size。
'''
class Expert(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        
        self.seq=nn.Sequential(
            nn.Linear(emb_size,emb_size),
            nn.ReLU(),
            nn.Linear(emb_size,emb_size),
        )

    def forward(self,x):
        return self.seq(x)

# Mixture of Experts
class MoE(nn.Module):
    def __init__(self,experts,top,emb_size,w_importance=0.01):
        super().__init__()
        # MoE需要包含n个Expert，这里构建了Expert对象列表。
        self.experts=nn.ModuleList([Expert(emb_size) for _ in range(experts)])
        # 保存top，用于选择top个Expert。
        self.top=top
        # 创建gate，它是一个输入是emb_size维度，输出是experts维度的线性变换。
        self.gate=nn.Linear(emb_size,experts)
        self.noise=nn.Linear(emb_size,experts)  # 给gate输出概率加噪音用
        self.w_importance=w_importance  # expert均衡用途(for loss)
        
    def forward(self,x):    # x: (batch,seq_len,emb)
        # 保存输入x的形状到x_shape变量，后面会用到。
        x_shape=x.shape
        
        # 为了兼容(batch, emb)和(batch, seq_len, emb)两种输入，
        # 将x的形状从(batch, seq_len, emb)变换为(batch*seqlen, emb)。
        x=x.reshape(-1,x_shape[-1]) # (batch*seq_len,emb)
        
        # gates 
        # 将x输入gate，得到每个Expert的logit。
        gate_logits=self.gate(x)    # (batch*seq_len,experts)
        # 将logit通过softmax，得到每个Expert的选择概率。
        gate_prob=softmax(gate_logits,dim=-1)   # (batch*seq_len,experts)
        
        # 为gate_prob添加噪声，优化expert倾斜问题。注意噪声只在训练时添加，推理时不会添加噪声。
        # 2024-05-05 Noisy Top-K Gating，优化expert倾斜问题
        if self.training: # 仅训练时添加噪音
            noise=torch.randn_like(gate_prob)*nn.functional.softplus(self.noise(x)) # https://arxiv.org/pdf/1701.06538 , StandardNormal()*Softplus((x*W_noise))
            gate_prob=gate_prob+noise
        
        # top expert
        # 调用PyTorch的topk算子，从gate_prob中找出最高的2个概率值，分别保存概率值和索引到top_weights和top_index中。
        # top_index用于选择执行哪个Expert，top_weights用于后面和推理结果做乘法。
        top_weights,top_index=torch.topk(gate_prob,k=self.top,dim=-1)   # top_weights: (batch*seq_len,top), top_index: (batch*seq_len,top)
        # 对选择出来的top_weights再做一次softmax操作，让top_weights成为相加和为1的概率值，每个值代表在每个选出来的expert的概率。
        top_weights=softmax(top_weights,dim=-1)
        
        top_weights=top_weights.view(-1)    # (batch*seq_len*top)
        top_index=top_index.view(-1)    # (batch*seq_len*top)
        
        # 因为每个emb要top个（这里以2个为例）Expert处理，所以将每个emb复制一份，变成2个。
        # 这一行执行完后，每个emb会出现2遍，之后才是下一个emb。2份emb，分别让2个Expert处理。
        x=x.unsqueeze(1).expand(x.size(0),self.top,x.size(-1)).reshape(-1,x.size(-1)) # (batch*seq_len*top,emb)
        # 定义与x相同形状的y，用于保存对x的处理结果。
        y=torch.zeros_like(x)   # (batch*seq_len*top,emb)
        
        '''
        注意这里的逻辑，按Expert来处理输入x，而不是按x来找Expert。
        也就是说，一共有8个Expert，我们遍历这8个Expert，对于Expert 1，我们找出所有需要Expert 1处理的x，让Expert 1统一处理。
        '''
        # run by per expert
        for expert_i,expert_model in enumerate(self.experts):
            # 筛选出所有需要expert_i处理的emb。
            x_expert=x[top_index==expert_i] # (...,emb)
            # 由expert_model一次性处理所有筛选出来的emb。
            y_expert=expert_model(x_expert)   # (...,emb)
            
            # 找到需要保存的位置。
            add_index=(top_index==expert_i).nonzero().flatten() # 要修改的下标
            # 保存处理结果到y。
            y=y.index_add(dim=0,index=add_index,source=y_expert)   # 等价于y[top_index==expert_i]=y_expert，为了保证计算图正确，保守用index_add算子
        
        # weighted sum experts
        # 对top_weights的形状做一些处理，使top_weights可以和y做乘法。
        top_weights=top_weights.view(-1,1).expand(-1,x.size(-1))  # (batch*seq_len*top,emb)
        # y和top_weights做乘法，即每个Expert的输出乘以该Expert对应的概率。
        y=y*top_weights
        # 改变y的形状，以让2个Expert的处理结果做加法。
        y=y.view(-1,self.top,x.size(-1))    # (batch*seq_len,top,emb)
        # 2个Expert处理结果做加法。
        y=y.sum(dim=1)  # (batch*seq_len,emb)
        
        # 用于均衡loss，避免expert倾斜。
        # 2024-05-05 计算gate输出各expert的累计概率, 做一个loss让各累计概率尽量均衡，避免expert倾斜
        # https://arxiv.org/pdf/1701.06538 BALANCING EXPERT UTILIZATION
        if self.training:
            importance=gate_prob.sum(dim=0) # 将各expert打分各自求和 sum( (batch*seq_len,experts) , dim=0)
            # 求CV变异系数（也就是让expert们的概率差异变小）, CV=标准差/平均值
            importance_loss=self.w_importance*(torch.std(importance)/torch.mean(importance))**2
        else:
            importance_loss=None 
        
        # 将y恢复x_shape变量保存的形状返回，同时返回gate_prob和importance_loss。
        return y.view(x_shape),gate_prob,importance_loss   # 2024-05-05 返回gate的输出用于debug其均衡效果, 返回均衡loss 

# MNIST分类
class MNIST_MoE(nn.Module):
    # MNIST_MoE模型包括一个embedding层，一个MoE层，一个classify层。
    def __init__(self,input_size,experts,top,emb_size):
        super().__init__()
        self.emb=nn.Linear(input_size,emb_size)
        self.moe=MoE(experts,top,emb_size)
        self.cls=nn.Linear(emb_size,10)
        
    def forward(self,x):
        x=x.view(-1,784)  # 将(28, 28)的图像变成(1, 784)的形状。
        y=self.emb(x)  # 做embedding。
        y,gate_prob,importance_loss=self.moe(y)  # MoE层处理。
        return self.cls(y),gate_prob,importance_loss  # classify层处理，同时返回gate_prob和importance_loss。

if __name__=='__main__':
    moe=MoE(experts=8,top=2,emb_size=16)
    x=torch.rand((5,10,16))
    y,prob,imp_loss=moe(x)
    print(y.shape,prob.shape,imp_loss.shape)
    
    mnist_moe=MNIST_MoE(input_size=784,experts=8,top=2,emb_size=16)
    x=torch.rand((5,1,28,28))
    y,prob,imp_loss=mnist_moe(x)
    print(y.shape)