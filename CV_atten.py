import torch, math, collections
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_geometric.nn as gn
from torchinfo import summary
import torch.optim as optim
from my_Utils import OUT_nodes,CFG


# 自注意力
class SelfAttention(nn.Module):
    def __init__(self,embedding_dim, in_channel):
        super(SelfAttention, self).__init__()
        # self.norm = nn.LayerNorm([embedding_dim, in_channel]).cuda()
        self.query = nn.Linear(in_channel, in_channel).cuda()
        self.key = nn.Linear(in_channel, in_channel).cuda()
        self.value = nn.Linear(in_channel, embedding_dim).cuda()
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Dropout(0.5)
    def forward(self, x):
        # x = self.norm(x)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = torch.matmul(Q, K.transpose(1, 2))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, V)
        return self.out(self.sigmoid(attended_values))

# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        # 输入参数为 单词embedding的维度 em_dim，MLP的缩放比例ratio
        self.Norm = nn.LayerNorm([1502, 128]) # 定义归一化
        self.GAP = nn.AdaptiveAvgPool1d(1) # 全局平均池化
        self.GMP = nn.AdaptiveMaxPool1d(1) # 全局最大池化
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1),
        ) # 定义share MLP
        self.Sigmoid = nn.Sigmoid() # 定义激活函数，用于将attention_feature转换为 score
        self.OUT = nn.Dropout(0.5) # 定义输出，用于summary显示

    def forward(self, x):
        x = self.Norm(x.permute(0, 2, 1))
        avg_out = self.GAP(x).squeeze(-1) # 压缩维度用于输入MLP
        max_out = self.GMP(x).squeeze(-1)
        avg_out = self.fc(avg_out).unsqueeze(2)
        max_out = self.fc(max_out).unsqueeze(2)
        out = avg_out + max_out # 分数求和
        attention_weights = self.Sigmoid(out)
        return self.OUT(attention_weights * x)

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, em_dim, seq_len, reduction_ratio=8):
        super(SpatialAttention, self).__init__()
        # 输入参数为句子长度seq_len，MLP的缩放比例ratio
        self.Norm = nn.LayerNorm([em_dim, seq_len]) # 定义归一化
        self.GAP = nn.AdaptiveAvgPool1d(1) # 全局平均池化
        self.GMP = nn.AdaptiveMaxPool1d(1) # 全局最大池化
        self.FC = nn.Sequential(
            nn.Linear(seq_len, seq_len // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(seq_len // reduction_ratio, seq_len),
            nn.LayerNorm([seq_len]) # 定义归一化
        ) # 定义MLP
        self.Sigmoid = nn.Sigmoid() # 定义激活函数，用于将attention_feature转换为 score
        self.OUT = nn.Dropout(0.5) # 定义输出，用于summary显示

    def forward(self, x):
        input = x
        x = self.Norm(x)
        x = x.permute(0, 2, 1) # 转换维度为 batch, seq_len, em_dim 用于压缩句子
        avg_out = self.GAP(x)
        max_out = self.GMP(x)
        x = avg_out + max_out
        weights = self.FC(x.permute(0, 2, 1)) # 还原维度为 batch, em_dim, seq_len
        attention_weights = self.Sigmoid(weights)
        return self.OUT(attention_weights * input)

# SKnet
class SelectiveKernelAttention(nn.Module):
    def __init__(self, em_dim, seq_len, Lkernel=16, Skernel=8, reduction_ratio=16):
        super(SelectiveKernelAttention, self).__init__()
        # 输入参数为 单词embedding的维度 em_dim 大卷积核大小 Lkernel 小卷积核大小 Skernel MLP的缩放比例ratio，
        # reduction_ratio是控制分支之间通道数压缩比例
        self.Norm = nn.BatchNorm1d(em_dim) # 定义归一化
        self.LargeKernel = Lkernel
        self.SmallKernel = Skernel
        self.LargeConv = nn.Sequential(
            nn.Conv1d(em_dim, em_dim, kernel_size=self.LargeKernel,
                      padding=(self.LargeKernel-1) // 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm([em_dim, seq_len-1]) # 定义归一化
        ) # 定义大尺度卷积
        self.SmallConv = nn.Sequential(
            nn.Conv1d(em_dim, em_dim, kernel_size=self.SmallKernel,
                      padding=(self.SmallKernel-1) // 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm([em_dim, seq_len-1]) # 定义归一化
        ) # 定义小尺度卷积
        self.GAP = nn.AdaptiveAvgPool1d(1) # 全局平均池化
        self.GMP = nn.AdaptiveMaxPool1d(1) # 全局最大池化
        self.FC = nn.Sequential(
            nn.Linear(em_dim, em_dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(em_dim // reduction_ratio, em_dim),
            nn.LayerNorm([em_dim]) # 定义归一化
        ) # 定义MLP
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
        # 输入参数为 句子长度seq_len, 单词embedding维度em_dim
        self.SE = ChannelAttention(em_dim, seq_len) # 定义通道注意力
        self.SP = SpatialAttention(em_dim, seq_len) # 定义空间注意力
        self.OUT = nn.Dropout(0.5) # 定义输出，用于summary显示
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



