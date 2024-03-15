from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, torch
from config import get_config
from utils import OUT_nodes, get_annotation_vec


args = get_config()

class MSData(Dataset):
    def __init__(self, benchmark_list, func='MF'):
        self.benchmark_list = benchmark_list
        self.annotations = {}
        self.esm_dir = args.esm_dir
        self.max_domain_len = args.max_domain_len
        self.max_seq_len = args.max_seq_len
        self.min_seq_len = args.min_seq_len

        self.esm_dict = {file.split('.pt')[0]: os.path.join(self.esm_dir, file) for file in
                         os.listdir(self.esm_dir)}

        self.sequences = {prot: seq for prot, seq in (line.strip().split(',') for line in open(args.data_sequence))
                          if self.min_seq_len <= len(seq) <= self.max_seq_len}

        self.domain_dict = {items[0]: list(map(int, items[1:])) for line in open(args.domain_file) for items in
                       [line.strip().split(',')] if items}

        self.ppiVecs = {items[0]: items[1:] for num, line in enumerate(open(args.ppi_file)) for items in
                        [line.strip().split(',')] if num != 0}

        self.annotations = get_annotation_vec(func=func,anno_file=args.anno_file)

    def __getitem__(self, idx):
        iprID = self.benchmark_list[idx]
        # Get the sequence feature vector
        esm_embed = [torch.load(self.esm_dict[iprID])['mean_representations'][33]]
        esm_feat = esm_embed[0]

        # Get the domain feature vector
        domain_s = self.domain_dict[iprID]
        if len(domain_s) >= self.max_domain_len:
            domain_s = np.array(domain_s[0:self.max_domain_len], dtype=int)
        else:
            domain_s = np.array(domain_s, dtype=int)
            domain_s = np.pad(domain_s, ((0, self.max_domain_len-len(domain_s))), 'constant', constant_values=0)
        domain_feat = torch.from_numpy(domain_s).type(torch.LongTensor).cuda()

        # get the ppi feature vector
        if iprID not in self.ppiVecs:
            ppiVect = np.zeros(1024, dtype=np.float_).tolist()
        else:
            ppiVect = self.ppiVecs[iprID]
            ppiVect = [float(x) for x in ppiVect]
        ppiVect = torch.Tensor(ppiVect).cuda()
        ppi_feat = ppiVect.type(torch.FloatTensor)

        # get idx label
        labels = self.annotations[iprID]
        labels = [int(x) for x in labels]
        labels = torch.Tensor(labels).cuda()
        labels = labels.type(torch.FloatTensor)
        labels = torch.squeeze(labels)

        return esm_feat, domain_feat, ppi_feat , labels

    def __len__(self):
        return len(self.benchmark_list)     #返回蛋白质的数量


class WeightData(Dataset):
    def __init__(self, benchmark_list, seq_dict, domain_dict, ppi_dict, func = 'MF'):
        self.benchmark = benchmark_list
        self.weight_dict = {}
        self.annotations = {}
        for i in range(len(benchmark_list)):
            prot = benchmark_list[i]
            temp = [seq_dict[prot], domain_dict[prot], ppi_dict[prot]]
            temp = np.array(temp)
            self.weight_dict[benchmark_list[i]] = temp.flatten().tolist()
            assert len(seq_dict[prot]) == len(domain_dict[prot]) == len(ppi_dict[prot]) == OUT_nodes[func]

        self.annotations = get_annotation_vec(func=func,anno_file=args.anno_file)

    def __getitem__(self, idx):
        prot = self.benchmark[idx]
        # Gets the input vector for weight_classifier
        weight_features = self.weight_dict[prot]
        weight_features = [float(x) for x in weight_features]
        weight_features = torch.Tensor(weight_features).cuda()
        weight_features = weight_features.type(torch.FloatTensor)
        # Get the GO label vector for the protein
        labels = self.annotations[prot]
        labels = [int(x) for x in labels]
        labels = torch.Tensor(labels).cuda()
        labels = labels.type(torch.FloatTensor)

        return weight_features, labels

    def __len__(self):
        return len(self.benchmark)


def read_benchmark():
    benchmark_file = args.benchmark_list
    all_data = []
    with open(benchmark_file, 'r') as f:
        for line in f:
            item = line.strip()
            all_data.append(item)
    return all_data
