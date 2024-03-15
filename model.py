import torch.nn as nn
import torch.nn.functional as F
from utils import CFG, OUT_nodes
import math, torch



class SeqLayer(nn.Module):
    def __init__(self, em_dim, b=1, gamma=2):
        super(SeqLayer, self).__init__()
        kernel_size = int(abs((math.log)(em_dim, 2) + b) / gamma)
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.GMP = nn.AdaptiveMaxPool1d(1)
        self.Conv1D = nn.Conv1d(2, 1, kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2)
        self.Sigmoid = nn.Sigmoid()
        self.OUT = nn.Dropout(0.3)

    def forward(self, x):
        input = x
        avg_out = self.GAP(x)
        max_out = self.GMP(x)
        x = torch.cat((avg_out, max_out), 2)
        x = self.Conv1D(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.Sigmoid(x)
        x = self.OUT(input * x)
        return x

# sequence model
class Seq_Module(nn.Module):
    def __init__(self, func):
        super(Seq_Module, self).__init__()
        self.Norm = nn.BatchNorm1d(1)
        self.eca = SeqLayer(128, 1280)
        self.seq_CNN = self.Multi_Conv(CFG['cfg05'])
        self.seq_FClayer = nn.Linear(2560, 1024)
        self.seq_outlayer = nn.Linear(1024, OUT_nodes[func])

    def forward(self, esm_feat):
        seq_out = esm_feat.unsqueeze(1)
        seq_out = self.Norm(seq_out)
        seq_out = self.eca(seq_out)
        seq_out = self.seq_CNN(seq_out)
        seq_out = seq_out.view(seq_out.size(0), -1)  # 展平
        seq_out = F.dropout(self.seq_FClayer(seq_out), p=0.5, training=self.training)
        seq_out = F.relu(seq_out)
        seq_out = self.seq_outlayer(seq_out)
        seq_out = F.sigmoid(seq_out)
        return seq_out

    def Multi_Conv(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2)]
            elif x == 'M2':
                layers += [nn.MaxPool1d(kernel_size=2, stride=1)]
            elif x == 'A':
                layers += [nn.AdaptiveAvgPool1d(1)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=16, stride=1, padding=8),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


class DomainLayer(nn.Module):
    def __init__(self, em_dim, seq_len, Lkernel=16, Skernel=8, reduction_ratio=16):
        super(DomainLayer, self).__init__()
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
        Large_score = self.Sigmoid(score)
        Small_score = 1 - Large_score
        Lout = Lx * Large_score
        Sout = Sx * Small_score
        out = self.OUT(Lout + Sout)
        return out


class Domain_Module(nn.Module):
    def __init__(self, func):
        super(Domain_Module, self).__init__()
        self.Dom_EmLayer = nn.Embedding(14243, 128, padding_idx=0).cuda()   # 嵌入数量，嵌入维度
        self.layer = DomainLayer(357,128)
        self.Dom_CNN = self.Multi_Conv(CFG['cfg07']).cuda()
        self.Dom_FClayer = nn.Linear(1088, 512).cuda()
        self.Dom_Outlayer = nn.Linear(512, OUT_nodes[func]).cuda()

    def forward(self, domain_feat): #seq 4981*100  ,domain
        domain_out = self.Dom_EmLayer(domain_feat)    # 32,357,128
        domain_out = self.layer(domain_out)
        domain_out = self.Dom_CNN(domain_out)
        domain_out = domain_out.view(domain_out.size(0), -1)  # 展平多维的卷积图
        domain_out = F.dropout(self.Dom_FClayer(domain_out), p=0.3, training=self.training)
        domain_out = F.relu(domain_out)
        domain_out = self.Dom_Outlayer(domain_out)
        domain_out = F.sigmoid(domain_out)
        return domain_out
    def Multi_Conv(self, cfg):
        layers = []
        in_channels = 357
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            elif x == 'M2':
                layers += [nn.MaxPool1d(kernel_size=2, stride=1)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=2, stride=2, padding=2),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


class PPI_Module(nn.Module):
    def __init__(self, func):
        super(PPI_Module, self).__init__()
        self.ppi_conv1d = nn.Conv1d(1, 8, kernel_size=4).cuda()
        self.ppi_hiddenlayer = nn.Linear(8168, 2048).cuda()
        self.ppi_outlayer = nn.Linear(2048, OUT_nodes[func]).cuda()

    def forward(self, ppi_feat):
        ppi_out = self.ppi_conv1d(ppi_feat.unsqueeze(1))
        ppi_out = F.relu(ppi_out)
        ppi_out = ppi_out.view(ppi_out.size(0), -1)
        ppi_out = F.dropout(self.ppi_hiddenlayer(ppi_out), p=0.3, training=self.training)
        ppi_out = F.relu(ppi_out)
        ppi_out = self.ppi_outlayer(ppi_out)
        ppi_out = F.sigmoid(ppi_out)
        return ppi_out

class Weight_classifier(nn.Module):
    def __init__(self, func):
        super(Weight_classifier, self).__init__()
        self.weight_layer = nn.Linear(OUT_nodes[func]*3, OUT_nodes[func])
        self.out= nn.Linear(OUT_nodes[func], OUT_nodes[func])

    def forward(self, weight_features):
        weight_out = self.weight_layer(weight_features)
        weight_out = F.relu(weight_out)
        weight_out = F.sigmoid(self.out(weight_out))
        return weight_out
