import torch,math
import torch.nn as nn
import torch.nn.functional as F
from my_Utils import OUT_nodes,CFG


class Seq_Module(nn.Module):
    def __init__(self, func,em_dim, b=1, gamma=2):
        super(Seq_Module, self).__init__()
        self.EmSeq = nn.Embedding(num_embeddings=25, embedding_dim=128, padding_idx=0).cuda()
        kernel_size = int(abs((math.log)(em_dim, 2) + b) / gamma)
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.GMP = nn.AdaptiveMaxPool1d(1)
        self.Conv1D = nn.Conv1d(2, 1, kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2)
        self.Sigmoid = nn.Sigmoid()
        self.OUT = nn.Dropout(0.3)
        self.seq_CNN = self.SeqConv1d(CFG['cfg05']).cuda()
        self.seq_FClayer = nn.Linear(3008, 1024).cuda()
        self.seq_outlayer = nn.Linear(1024, OUT_nodes[func]).cuda()

    def forward(self, seqMatrix):
        seqMatrix= self.EmSeq(seqMatrix) 
        seqMatrix = seqMatrix.permute(0, 2, 1) 
        input = x
        avg_out = self.GAP(x)
        max_out = self.GMP(x)
        x = torch.cat((avg_out, max_out), 2)
        x = self.Conv1D(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.Sigmoid(x)
        x = self.OUT(input * x)
        seq_out = self.seq_CNN(seqMatrix)
        seq_out = seq_out.view(seq_out.size(0), -1)
        seq_out = F.dropout(self.seq_FClayer(seq_out), p=0.5, training=self.training)
        seq_out = F.relu(seq_out)
        seq_out = self.seq_outlayer(seq_out)
        seq_out = F.sigmoid(seq_out)
        return seq_out
        

    def SeqConv1d(self, cfg):
        layers = []
        in_channels = 128
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

