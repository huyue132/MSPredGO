
import torch.nn as nn
import torch.nn.functional as F
from my_Utils import OUT_nodes


class PPI_Module(nn.Module):
    def __init__(self, func):
        super(PPI_Module, self).__init__()
        self.ppi_conv1d = nn.Conv1d(1, 2, kernel_size=4).cuda()
        self.ppi_hiddenlayer = nn.Linear(32762, 2048).cuda()
        self.ppi_outlayer = nn.Linear(2048, OUT_nodes[func]).cuda()

    def forward(self, ppiVec):
        ppi_out = self.ppi_conv1d(ppiVec.unsqueeze(1))
        ppi_out = F.relu(ppi_out)
        ppi_out = ppi_out.view(ppi_out.size(0), -1)
        ppi_out = F.dropout(self.ppi_hiddenlayer(ppi_out), p=0.3, training=self.training)
        ppi_out = F.relu(ppi_out)
        ppi_out = self.ppi_outlayer(ppi_out)
        ppi_out = F.sigmoid(ppi_out)
        return ppi_out