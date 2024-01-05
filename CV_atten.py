import torch, math, collections
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_geometric.nn as gn
from torchinfo import summary
import torch.optim as optim
from my_Utils import OUT_nodes,CFG


class SelfAttention(nn.Module):
    def __init__(self,embedding_dim, in_channel):
        super(SelfAttention, self).__init__()
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

# 空间注意力
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
