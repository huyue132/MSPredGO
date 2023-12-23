# coding: utf-8
# some tools
# import pandas as pd
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics
import torch
import torch.nn as nn
import math
from einops import rearrange,repeat
CFG = {
    'cfg00': [16, 'M', 16, 'M'],
    'cfg01': [16, 'M', 32, 'M'],
    'cfg02': [32, 'M'],
    'cfg03': [64, 'M'],
    'cfg04': [16, 'M', 16, 'M',32, 'M'],
    'cfg05': [64, 'M', 32, 'M',16, 'M'],
    'cfg06': [64, 'M', 32, 'M',32, 'M'],
    'cfg07': [128, 'M', 64, 'M2'],
    'cfg08': [512, 'M', 128, 'M2',32, 'M2'],
    'cfg09': [128, 'M', 64, 'M'],
    'cfg10': [64, 'M'],
    'cfg11': [32, 'M'],
    'cfg12': [512, 'M2', 128, 'M2', 32, 'M']
}
#使用一组阈值（BP为40，MF为20，CC为20）选择GO术语，人类蛋白质包含491个BP项、321个MF项和240个CC项
OUT_nodes = {
    'BP': 491,
    'MF': 321,
    'CC': 240,
}

Thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
              0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
              0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
              0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,
              0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
              0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
              0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,
              0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
              0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
              0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

def benchmark_set_split(term_arg='MF'):
    benchmark_file = 'data/{}_benchmarkSet_2.csv'.format(term_arg)
    print(benchmark_file)
    trainset, testset = [], []
    all_data = []
    with open(benchmark_file, 'r') as f:
        for line in f:
            item = line.strip()
            all_data.append(item)
    idx_list = np.arange(len(all_data)).tolist()
    nums = {
        'BP': 10700,
        'MF': 10500,
        'CC': 10000,
        'test': 10
    }
    random_index = []
    with open('data/{}_random_index.csv'.format(term_arg), 'r') as f:
        for line in f:
            item = line.strip().split(',')
            random_index.append(int(item[0]))
    for i in range(len(all_data)):
        if i in random_index:
            trainset.append(all_data[i])
        else:
            testset.append(all_data[i])
    assert len(trainset) + len(testset) == len(all_data)
    return trainset, testset

def calculate_performance(actual, pred_prob, threshold=0.4, average='micro'):
    pred_lable = []
    for l in range(len(pred_prob)):
        eachline = (np.array(pred_prob[l]) > threshold).astype(np.int_)
        eachline = eachline.tolist()
        pred_lable.append(eachline)
    f_score = f1_score(np.array(actual), np.array(pred_lable), average=average)
    recall = recall_score(np.array(actual), np.array(pred_lable), average=average)
    precision = precision_score(np.array(actual), np.array(pred_lable), average=average)
    return f_score, recall,  precision

def cacul_aupr(lables, pred):
    precision, recall, _thresholds = metrics.precision_recall_curve(lables, pred)
    aupr = metrics.auc(recall, precision)
    return aupr


# 通过序列，获取该序列的k_mers矩阵
def mer_k(seq, protvec, k = 3, file_base = 'data/3_mers_base.csv'): #3-mers
    three_mer = []
    with open(file_base, 'r') as f:
        for line in f:
            three_mer.append(str(line.strip())) #将3_mers_base.csv文件中的组合放到three_mer列表中
    protVec = protvec
    zeroVec = np.zeros(100, dtype= float).tolist()  #返回一个给定形状和类型的用0填充的数组l
    # zeroVec = np.zeros(8000, dtype=float).tolist()
    l = []
    seq_length = len(seq)
    for i in range(seq_length):
        t = seq[i:(i + k)]
        if (len(t)) == k:
            if t in three_mer:
                vec = protVec[t]
                l.append(vec)
            else:
                l.append(zeroVec)
            # break
    return l

# 通过序列，获取该序列的k_mers的Sentence
def mer_k_Sentence(seq, protdict, k = 3, file_base = 'data/3_mers_base.csv'):
    three_mer = []
    with open(file_base, 'r') as f:
        for line in f:
            three_mer.append(str(line.strip()))
    protDict = protdict
    # zeroVec = np.zeros(100, dtype= float).tolist()
    # zeroVec = np.zeros(8000, dtype=float).tolist()
    l = []
    seq_length = len(seq)
    for i in range(seq_length):
        t = seq[i:(i + k)]
        if (len(t)) == k:
            if t in three_mer:
                word = protDict[t]
                l.append(word)
            else:
                l.append(0)
            # break
    return l

# 通过domain，获取该protein的domain矩阵
def get_domain_matrix(domain_s, domainVec):
    l =[]
    for i in range(len(domain_s)):
        if domain_s[i] in domainVec:
            vec = domainVec[domain_s[i]]
            l.append(vec)
        else:
            # zero_vec = np.zeros((128), dtype=np.float).tolist()
            zero_vec = np.zeros((14242), dtype=np.float).tolist()
            l.append(zero_vec)
    return l

def make_k_mers_base():
    Amino_acid = ['A','C','D','E','F','G','H','I','K','L',
                  'M','N','P','Q','R','S','T','V','W','Y']
    k_mers = []
    for i in range(len(Amino_acid)):
        for j in range(len(Amino_acid)):
            for k in range(len(Amino_acid)):
                each_mer = Amino_acid[i] + Amino_acid[j] + Amino_acid[k]
                k_mers.append(each_mer)
    assert len(k_mers) == 8000
    # with open("data/3_mers_base.csv", 'w') as f:
    #     for line in range(len(k_mers)):
    #         f.write('{}\n'.format(str(k_mers[line])))
    return k_mers

def creat_kmer_metrix(k=3):
    kmers_base = make_k_mers_base()
    # zerolist = [0 for x in range(len(kmers_base))]
    kmers_dict = {}
    for i in range(len(kmers_base)):
        # temp = zerolist
        # temp[i] = 1
        kmers_dict[kmers_base[i]] = [0 for x in range(len(kmers_base))]
        kmers_dict[kmers_base[i]][i] = 1
    # print(kmers_dict['AAA'],kmers_dict['YYY'] )
    np.save('data/prot_onehot_dict.npy', kmers_dict)

def k_mer(seq,k=1):
    sentences = []
    for i in range(len(seq)-k +1):
        kmer = seq[i:i+k]
        sentences.append(kmer)
    return sentences

def creat_kmer_Wordict(k=3):
    kmers_base = make_k_mers_base()
    # zerolist = [0 for x in range(len(kmers_base))]
    kmers_dict = {}
    for i in range(len(kmers_base)):
        # temp = zerolist
        # temp[i] = 1
        kmers_dict[kmers_base[i]] = i + 1
    # print(kmers_dict['AAA'],kmers_dict['YYY'] )
    np.save('data/prot_kmerWord_dict.npy', kmers_dict)
    with open('data/prot3mersWordict.csv', 'w') as f:
        for j in range(len(kmers_base)):
            f.write('{},'.format(kmers_base[j]))
            f.write('{}\n'.format(kmers_dict[kmers_base[j]]))

class SelfAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        d = x.shape[-1]
        queries = x
        keys = x
        values = x
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = torch.softmax(scores, dim=2)
        return torch.bmm(self.dropout(self.attention_weights), values)

#定义位置编码
class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.is_absolute = True
        self.embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer('position_ids', torch.arange(max_position_embeddings))

    def forward(self, x):

        """
        return (b l d) / (b h l d)
        """
        position_ids = self.position_ids[:x.size(-2)]

        if x.dim() == 3:
            return x + self.embeddings(position_ids)[None, :, :]

        elif x.dim() == 4:
            h = x.size(1)
            x = rearrange(x, 'b h l d -> b l (h d)')
            x = x + self.embeddings(position_ids)[None, :, :]
            x = rearrange(x, 'b l (h d) -> b h l d', h=h)
            return x

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

# creat_kmer_Wordict()
# creat_kmer_metrix()
# print(done)
# if __name__ == '__main__':
#     mer_k()