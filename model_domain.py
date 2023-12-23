
import torch.nn as nn
import torch.nn.functional as F
from my_Utils import OUT_nodes,CFG


class Domain_Module(nn.Module):
    def __init__(self, func, features=357,em_dim=128,Lkernel = 16, Skernel=8, reduction_ratio = 16):
        super(Domain_Module, self).__init__()
        self.LargeKernel = Lkernel
        self.SmallKernel = Skernel
        self.Dom_EmLayer = nn.Embedding(14243, 128, padding_idx=0).cuda()   
        self.LargeConv = nn.Sequential(
            nn.Conv1d(features, features, kernel_size=self.LargeKernel,
                      padding=(self.LargeKernel-1) // 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm([features, em_dim-1])
        )
        self.SmallConv = nn.Sequential(
            nn.Conv1d(features, features, kernel_size=self.SmallKernel,
                      padding=(self.SmallKernel-1) // 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm([features,em_dim-1])
        )
        self.FC = nn.Sequential(
            nn.Linear(features, features // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(features // reduction_ratio, features),
            nn.LayerNorm([features])
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
        self.Dom_CNN = self.DomainConv1d(CFG['cfg07']).cuda()
        self.Dom_FClayer = nn.Linear(1088, 512).cuda()
        self.Dom_Outlayer = nn.Linear(512, OUT_nodes[func]).cuda()

    def forward(self, domainSentence): 
        x = self.Dom_EmLayer(domainSentence)    
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
        domain_out = self.OUT(Lout + Sout)
        domain_out = self.Sk(domain_out)
        domain_out = self.Dom_CNN(domain_out)
        domain_out = domain_out.view(domain_out.size(0), -1)  
        domain_out = F.dropout(self.Dom_FClayer(domain_out), p=0.3, training=self.training)
        domain_out = F.relu(domain_out)
        domain_out = self.Dom_Outlayer(domain_out)
        domain_out = F.sigmoid(domain_out)
        return domain_out

        
    def DomainConv1d(self, cfg):
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
