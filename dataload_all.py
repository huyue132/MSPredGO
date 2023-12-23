import numpy as np
import torch
from torch.utils.data import Dataset
from my_Utils import OUT_nodes

ProtVec = np.load('data/protVec_dict.npy',allow_pickle=True).item()
Seqfile_name = 'data/seqSet.csv'
Domainfile_name = 'data/NewdomainSet.csv'
GOfile_name = 'data/humanProteinGO.csv'

from transformers import PreTrainedTokenizerFast
model = "data/model/tokenizer.json"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model)

class Dataload(Dataset):
    def __init__(self, benchmark_list, seqfile_name, domainfile_name, GOfile_name, func='MF', transform=None):
        self.benchmark_list = benchmark_list
        self.sequeces = {}
        self.max_seq_len = 1500  
        self.doamins = {}
        self.max_domains_len = 357
        self.ppiVecs = {}
        self.GO_annotiations = {}
        self.max_label_len = 70

        with open(seqfile_name, 'r') as f:  
            for line in f:
                items = line.strip().split(',')
                prot, seq = items[0], items[1]
                self.sequeces[prot] = seq
        self.protDict = ProtVec

        with open(domainfile_name, 'r') as f:   
            for line in f:
                items = line.strip().split(',')
                prot, domains = items[0], items[1:]
                domains = [int(x) for x in domains]
                self.doamins[prot] = domains

        ppi_file = 'data/ppi_PCA_512.csv'
        with open(ppi_file, 'r') as f:
            num = 1
            for line in f:
                if num == 1:
                    num = 2
                    continue
                items = line.strip().split(',')
                prot, vector = items[0], items[1:]
                self.ppiVecs[prot] = vector

        with open(GOfile_name, 'r') as f:       
            num = 1
            for line in f:
                if num == 1:
                    num = 2
                    continue
                items = line.strip().split(',')
                if func == 'BP':
                    prot, GO_annotiation = items[0], items[1:492]
                elif func == 'MF':
                    prot, GO_annotiation = items[0], items[492:813]
                elif func == 'CC':
                    prot, GO_annotiation = items[0], items[813:]
                self.GO_annotiations[prot] = GO_annotiation

    def __getitem__(self, idx):
        iprID = self.benchmark_list[idx]
        
        seq = self.sequeces[iprID]
        if len(seq) > self.max_seq_len:
            seq = seq[0:self.max_seq_len]
        seqMatrix = seq
        seqMatrix = "[SOS]" + seqMatrix + "[EOS]"
        
        seqMatrix = tokenizer.tokenize(seqMatrix)
        
        seqMatrix = tokenizer.convert_tokens_to_ids(seqMatrix)
        seqMatrix = np.array(seqMatrix)
        seqMatrix = np.pad(seqMatrix, ((0, 1502 - len(seqMatrix))), 'constant', constant_values=0)
        seqMatrix = torch.from_numpy(seqMatrix).type(torch.LongTensor).cuda()
        seqMatrix = seqMatrix.type(torch.LongTensor)
        
        domain_s = self.doamins[iprID]
        if len(domain_s) >= self.max_domains_len:
            domain_s = np.array(domain_s[0:self.max_domains_len], dtype=int)
        else:
            domain_s = np.array(domain_s, dtype=int)
            domain_s = np.pad(domain_s, ((0, self.max_domains_len-len(domain_s))), 'constant', constant_values=0)
        domainSentence = torch.from_numpy(domain_s).type(torch.LongTensor).cuda()
        
        if iprID not in self.ppiVecs:
            ppiVect = np.zeros((512), dtype=np.float_).tolist()
        else:
            ppiVect = self.ppiVecs[iprID]
            ppiVect = [float(x) for x in ppiVect]
        ppiVect = torch.Tensor(ppiVect).cuda()
        ppiVect = ppiVect.type(torch.FloatTensor)

        
        GO_annotiations = self.GO_annotiations[iprID]
        GO_annotiations = [int(x) for x in GO_annotiations]
        GO_annotiations = torch.Tensor(GO_annotiations).cuda()
        GO_annotiations = GO_annotiations.type(torch.FloatTensor)


        return seqMatrix, domainSentence, ppiVect , GO_annotiations

    def __len__(self):
        return len(self.benchmark_list)     

class weight_Dataload(Dataset):
    def __init__(self, benchmark_list, seqdict, domaindict, ppidict, GOfile_name, func = 'MF', transform=None):
        self.benchmark = benchmark_list
        self.weghtdict = {}
        self.GO_annotiations = {}

        for i in range(len(benchmark_list)):
            prot = benchmark_list[i]
            
            
            temp = [seqdict[prot], domaindict[prot], ppidict[prot]]
            temp = np.array(temp)
            self.weghtdict[benchmark_list[i]] = temp.flatten().tolist()
            assert len(seqdict[prot]) == len(domaindict[prot]) == len(ppidict[prot]) == OUT_nodes[func]

        with open(GOfile_name, 'r') as f:       
            num = 1
            for line in f:
                if num == 1:
                    num = 2
                    
                    
                    continue
                items = line.strip().split(',')
                if func == 'BP':
                    prot, GO_annotiation = items[0], items[1:492]
                elif func == 'MF':
                    prot, GO_annotiation = items[0], items[492:813]
                elif func == 'CC':
                    prot, GO_annotiation = items[0], items[813:]
                
                self.GO_annotiations[prot] = GO_annotiation



    def __getitem__(self, idx):
        prot = self.benchmark[idx]

        
        weight_features = self.weghtdict[prot]
        weight_features = [float(x) for x in weight_features]
        weight_features = torch.Tensor(weight_features).cuda()
        weight_features = weight_features.type(torch.FloatTensor)


        
        GO_annotiations = self.GO_annotiations[prot]
        GO_annotiations = [int(x) for x in GO_annotiations]
        GO_annotiations = torch.Tensor(GO_annotiations).cuda()
        
        GO_annotiations = GO_annotiations.type(torch.FloatTensor)

        return weight_features, GO_annotiations

    def __len__(self):
        return len(self.benchmark)

