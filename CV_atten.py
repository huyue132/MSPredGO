import torch, math, collections
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_geometric.nn as gn
from torchinfo import summary
import torch.optim as optim
from my_Utils import OUT_nodes,CFG



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.Norm = nn.LayerNorm([1502, 128])
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.GMP = nn.AdaptiveMaxPool1d(1) 
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1),
        ) 
        self.Sigmoid = nn.Sigmoid() 
        self.OUT = nn.Dropout(0.5) 
        
    def forward(self, x):
        x = self.Norm(x.permute(0, 2, 1))
        avg_out = self.GAP(x).squeeze(-1) 
        max_out = self.GMP(x).squeeze(-1)
        avg_out = self.fc(avg_out).unsqueeze(2)
        max_out = self.fc(max_out).unsqueeze(2)
        out = avg_out + max_out 
        attention_weights = self.Sigmoid(out)
        return self.OUT(attention_weights * x)


class SpatialAttention(nn.Module):
    def __init__(self, em_dim, seq_len, reduction_ratio=8):
        super(SpatialAttention, self).__init__()
        self.Norm = nn.LayerNorm([em_dim, seq_len]) 
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.GMP = nn.AdaptiveMaxPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(seq_len, seq_len // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(seq_len // reduction_ratio, seq_len),
            nn.LayerNorm([seq_len]) 
        ) 
        self.Sigmoid = nn.Sigmoid()
        self.OUT = nn.Dropout(0.5)

    def forward(self, x):
        input = x
        x = self.Norm(x)
        x = x.permute(0, 2, 1) 
        avg_out = self.GAP(x)
        max_out = self.GMP(x)
        x = avg_out + max_out
        weights = self.FC(x.permute(0, 2, 1))
        attention_weights = self.Sigmoid(weights)
        return self.OUT(attention_weights * input)

class SelectiveKernelAttention(nn.Module):
    def __init__(self, em_dim, seq_len, Lkernel=16, Skernel=8, reduction_ratio=16):
        super(SelectiveKernelAttention, self).__init__()
        self.Norm = nn.BatchNorm1d(em_dim)
        self.LargeKernel = Lkernel
        self.SmallKernel = Skernel
        self.LargeConv = nn.Sequential(
            nn.Conv1d(em_dim, em_dim, kernel_size=self.LargeKernel,
                      padding=(self.LargeKernel-1) // 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm([em_dim, seq_len-1])
        )
        self.SmallConv = nn.Sequential(
            nn.Conv1d(em_dim, em_dim, kernel_size=self.SmallKernel,
                      padding=(self.SmallKernel-1) // 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm([em_dim, seq_len-1])
        )
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.GMP = nn.AdaptiveMaxPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(em_dim, em_dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(em_dim // reduction_ratio, em_dim),
            nn.LayerNorm([em_dim])
        ) 
        self.Sigmoid = nn.Sigmoid()
        self.OUT = nn.Dropout(0.6)

    def forward(self, x):
        x = self.Norm(x)
        Lx = self.LargeConv(x)
        Sx = self.SmallConv(x)
        x_unite = Lx + Sx
        avg_out = self.GAP(x_unite).squeeze(-1)
        max_out = self.GMP(x_unite).squeeze(-1)
        avg_out = self.FC(avg_out)
        max_out = self.FC(max_out)
        score = avg_out + max_out
        score = score.view(score.size(0), -1, 1)
        Largescore = self.Sigmoid(score)
        Smallscore = 1 - Largescore
        Lout = Lx * Largescore
        Sout = Sx * Smallscore
        out = self.OUT(Lout + Sout)
        return out

#CBAM
class CBAM_seq(nn.Module):
    def __init__(self, em_dim, seq_len,):
        super(CBAM_seq, self).__init__()
        self.SE = ChannelAttention(em_dim, seq_len) 
        self.SP = SpatialAttention(em_dim, seq_len) 
        self.OUT = nn.Dropout(0.5)
    def forward(self, x):
        x = self.SE(x)
        x = self.SP(x)
        x = self.OUT(x)
        return x

#ECA
class ECA(nn.Module):
    def __init__(self, em_dim, seq_len, b=1, gamma=2):
        super(ECA, self).__init__()
        # 输入参数为单词embedding维度 em_dim
        # self.Norm = nn.BatchNorm1d([em_dim, seq_len])  # 定义归一化
        # 用于自适应定义一维卷积kernel_size
        kernel_size = int(abs((math.log)(em_dim, 2) + b) / gamma)
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.GAP = nn.AdaptiveAvgPool1d(1) # 全局平均池化
        self.GMP = nn.AdaptiveMaxPool1d(1) # 全局最大池化
        self.Conv1D = nn.Conv1d(2, 1, kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2) # 定义代替全连接层的一维卷积
        self.Sigmoid = nn.Sigmoid() # 定义激活函数，用于将attention_feature转换为 score
        self.OUT = nn.Dropout(0.3) # 定义输出，用于summary显示

    def forward(self, x):
        input = x
        # x = self.Norm(x)
        avg_out = self.GAP(x)
        max_out = self.GMP(x)
        x = torch.cat((avg_out, max_out), 2)
        x = self.Conv1D(x.permute(0, 2, 1)).permute(0, 2, 1)
        # # max_out = self.Conv1D(max_out.permute(0, 2, 1)).permute(0, 2, 1)
        # # x = avg_out + max_out
        x = self.Sigmoid(x)
        # # 将维度调换，用于单通道卷积输入输出，卷积结束后再进行维度调换，以适应原输入
        x = self.OUT(input * x)
        return x



