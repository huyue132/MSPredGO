

from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import os,time,torch, warnings,math
from my_Utils import Thresholds,OUT_nodes,calculate_performance,cacul_aupr,CFG
from prettytable import PrettyTable
from torchinfo import summary
from transformers import PreTrainedTokenizerFast

warnings.filterwarnings("ignore")
torch.manual_seed(100)  
torch.cuda.seed()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
ProtVec = np.load('data/protVec_dict.npy',allow_pickle=True).item()
Seqfile_name = 'data/seqSet.csv'
Domainfile_name = 'data/NewdomainSet.csv'
GOfile_name = 'data/humanProteinGO.csv'
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

        ppi_file = 'data/ppi_PCA_1024.csv'
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
            ppiVect = np.zeros((1024), dtype=np.float_).tolist()
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
            if func == 'MF':

                weight_seq = 0.1  
                weight_domain = 0.8  
                weight_ppi = 0.1  
            else:
                weight_seq = 0.1  
                weight_domain = 0.1  
                weight_ppi = 0.8  
            weighted_average = (weight_seq * np.array(seqdict[prot]) +
                                weight_domain * np.array(domaindict[prot]) +
                                weight_ppi * np.array(ppidict[prot]))
            self.weghtdict[benchmark_list[i]] = weighted_average.tolist()
            assert len(weighted_average) == OUT_nodes[func]
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

class Weight_classifier(nn.Module):
    def __init__(self, func):
        super(Weight_classifier, self).__init__()
        self.weight_layer = nn.Linear(OUT_nodes[func], OUT_nodes[func])
        self.outlayer= nn.Linear(OUT_nodes[func], OUT_nodes[func])

    def forward(self, weight_features):
        weight_out = self.weight_layer(weight_features)
        weight_out = F.relu(weight_out)
        weight_out = F.sigmoid(self.outlayer(weight_out))
        return weight_out


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


def Main(train_benchmark, test_benchmark, func='MF'):
    if func == 'BP':
        seq_train_out, seq_test_out, seq_t = Seq_train(0.0001, 16, train_benchmark, test_benchmark, 30, func)  
        domain_train_out, domain_test_out, domain_t = Domain_train(0.001, 32, train_benchmark, test_benchmark, 45,
                                                                   func)  
    else:
        seq_train_out, seq_test_out, seq_t = Seq_train(0.001, 8, train_benchmark, test_benchmark, 35, func)  
        domain_train_out, domain_test_out, domain_t = Domain_train(0.001, 16, train_benchmark, test_benchmark, 35,
                                                                   func)  
    ppi_train_out, ppi_test_out, ppi_t = PPI_train(0.0001, 8, train_benchmark, test_benchmark, 40, func)  
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('{}  Weight_model start'.format(func))
    learning_rate = 0.001
    batch_size = 32
    epoch_times = 30
    weight_model = Weight_classifier(func).cuda()
    summary(weight_model, input_size=(32, OUT_nodes[func]))
    print('batch_size:{} learning_rate:{} epoch_times:{}'.format(batch_size, learning_rate, epoch_times))
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(weight_model.parameters(), lr=learning_rate, weight_decay=0.00001)

    train_dataset = weight_Dataload(train_benchmark, seq_train_out, domain_train_out, ppi_train_out, GOfile_name, func=func)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = weight_Dataload(test_benchmark, seq_test_out, domain_test_out, ppi_test_out, GOfile_name, func=func)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    print('\nWeightModel training begins')
    print("{:^7} {:^10} {:^10} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8}".format('Epoch', 'Loss', 'TestLoss', 'T', 'F_score',
                                                                           'AUC', 'AUPR', 'Recall', 'Prec', 'EpochTime'))
    print('-' * 95)
    wsince = time.time()
    best_fscore = 0
    for epoch in range(epoch_times):
        _loss = 0
        batch_num = 0

        for batch_idx, (weight_features, GO_annotiations) in enumerate(train_data_loader):
            weight_model.train()
            since = time.time()
            weight_features = Variable(weight_features).cuda()
            GO_annotiations = torch.squeeze(GO_annotiations)
            GO_annotiations = Variable(GO_annotiations).cuda()
            out = weight_model(weight_features)
            optimizer.zero_grad()
            loss = loss_function(out, GO_annotiations)
            batch_num += 1
            loss.backward()
            optimizer.step()
            _loss += loss.item()
        epoch_loss = "{}".format(_loss / batch_num)
        t_loss = 0
        test_batch_num = 0
        pred = []
        actual = []
        for idx, (weight_features, GO_annotiations) in enumerate(test_data_loader):
            weight_model.eval()
            weight_features = Variable(weight_features).cuda()
            GO_annotiations = Variable(GO_annotiations).cuda()
            out = weight_model(weight_features)
            test_batch_num = test_batch_num + 1
            pred.append(out.data[0].cpu().tolist())
            actual.append(GO_annotiations.data[0].cpu().tolist())
            one_loss = loss_function(out, GO_annotiations)
            t_loss += one_loss.item()
        test_loss = "{}".format(t_loss / test_batch_num)
        fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        aupr_score = cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())
        score_dict = {}
        each_best_fcore = 0
        each_best_scores = []
        for i in range(len(Thresholds)):
            f_score, recall, precision = calculate_performance(
                actual, pred, threshold=Thresholds[i], average='micro')
            if f_score >= each_best_fcore:
                each_best_fcore = f_score
                each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score, aupr_score]
            scores = [f_score, recall, precision, auc_score, aupr_score]
            score_dict[Thresholds[i]] = scores
        if each_best_fcore >= best_fscore:
            best_fscore = each_best_fcore
            best_scores = each_best_scores
            best_score_dict = score_dict
            torch.save(weight_model,
                       'savedpkl/WeightVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times))
        t, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
        precision, auc_score, aupr_score = each_best_scores[3], each_best_scores[4], each_best_scores[5]
        time_elapsed = time.time() - since
        epochtime = str(int(time_elapsed) // 60) + 'm' + ' ' + str(int(time_elapsed % 60)) + 's'
        epoch_loss = float(epoch_loss)
        test_loss = float(test_loss)
        f_score = float(f_score)
        t = float(t)
        auc_score = float(auc_score)
        aupr_score = float(aupr_score)
        recall = float(recall)
        precision = float(precision)
        epoch += 1
        print(format(format(epoch, '0>2d'), '^7'),
              format(format(epoch_loss, '.7f'), '^10'),
              format(format(test_loss, '.7f'), '^10'),
              format(format(t, '.3f'), '^8'),
              format(format(f_score, '.3f'), '^8'),
              format(format(auc_score, '.3f'), '^8'),
              format(format(aupr_score, '.3f'), '^8'),
              format(format(recall, '.3f'), '^8'),
              format(format(precision, '.3f'), '^8'),
              format(epochtime, '^8')
              )
    print('WeightModel training ends')
    bestthreshold, f_max, recall_max = best_scores[0], best_scores[1], best_scores[2]
    prec_max, bestauc_score, bestaupr_score = best_scores[3], best_scores[4], best_scores[5]
    print('\nWeightModel evaluation begins')
    w_elapsed = time.time() - wsince
    test_loss = float(test_loss)
    f_max = float(f_max)
    bestauc_score = float(bestauc_score)
    bestaupr_score = float(aupr_score)
    recall_max = float(recall_max)
    prec_max = float(prec_max)
    bestthreshold = float(bestthreshold)
    table = PrettyTable(['TestLoss','Lr','Batch','Epoch','Fmax','AUC','AUPR','Recall','Prec','T','Time',])
    modeltime = str(int(w_elapsed) // 60)+'m'+' '+str(int(w_elapsed % 60))+'s'
    table.add_row([format(test_loss,'.5f'),learning_rate,batch_size,epoch_times,format(f_max,'.3f'),
                   format(bestauc_score,'.3f'),format(bestaupr_score,'.3f'),format(recall_max,'.3f'), format(prec_max,'.3f'), format(bestthreshold,'.3f'), modeltime])
    print(table)

    test_weight_model = torch.load(
        'savedpkl/WeightVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times)).cuda()
    t_loss = 0
    weight_test_outs = {}
    pred = []
    actual = []
    score_dict = {}
    batch_num = 0
    for batch_idx, (weight_features, GO_annotiations) in enumerate(test_data_loader):
        weight_features = Variable(weight_features).cuda()
        GO_annotiations = Variable(GO_annotiations).cuda()
        out = test_weight_model(weight_features)
        batch_num += 1
        weight_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
        pred.append(out.data[0].cpu().tolist())
        actual.append(GO_annotiations.data[0].cpu().tolist())
        loss = loss_function(out, GO_annotiations)
        t_loss += loss.item()
    test_loss = "{}".format(t_loss / batch_num)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    aupr = cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())
    each_best_fcore = 0
    for i in range(len(Thresholds)):
        f_score, recall, precision = calculate_performance(
            actual, pred, threshold=Thresholds[i], average='micro')
        if f_score > each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score, aupr]
        scores = [f_score, recall, precision, auc_score]
        score_dict[Thresholds[i]] = scores
    print('WeightModel evaluation ends')
    return each_best_scores

def read_benchmark(term_arg='MF'):
    benchmark_file = 'data/{}_benchmarkSet_2.csv'.format(term_arg)
    all_data = []
    with open(benchmark_file, 'r') as f:
        for line in f:
            item = line.strip()
            all_data.append(item)
    return all_data

def validation(func='MF', k_fold=5):
    kf = KFold(n_splits=k_fold)
    benchmark = np.array(read_benchmark(func))
    scores = []
    for train_index, test_index in kf.split(benchmark):
        train_set = benchmark[train_index].tolist()
        test_set = benchmark[test_index].tolist()
        each_fold_scores = Main(train_set, test_set, func=func)
        scores.append(each_fold_scores)
    f_maxs, pre_maxs, rec_maxs, auc_s, aupr_s = [], [], [], [], []
    for i in range(len(scores)):
        f_maxs.append(scores[i][1])
        rec_maxs.append(scores[i][2])
        pre_maxs.append(scores[i][3])
        auc_s.append(scores[i][4])
        aupr_s.append(scores[i][5])
    f_mean = np.mean(np.array(f_maxs))
    rec_mean = np.mean(np.array(rec_maxs))
    pre_mean = np.mean(np.array(pre_maxs))
    auc_mean = np.mean(np.array(auc_s))
    aupr_mean = np.mean(np.array(aupr_s))
    return f_mean, rec_mean, pre_mean, auc_mean, aupr_mean


if __name__ == '__main__':
    time_start1 = time.time()
    Terms = ['BP']
    f_mean, rec_mean, pre_mean, auc_mean, aupr_mean = validation(Terms[0], 5)

    time_end = time.time() - time_start1
    alltime = str(int(time_end) // 60) + 'm' + ' ' + str(int(time_end % 60)) + 's'
    table1 = PrettyTable(['FUNC', 'Fmax_Mean', 'Rec_Mean', 'Prec_Mean', 'AUC_Mean', 'AUPR_Mean', 'AllTime'])
    table1.add_row([Terms[0], format(f_mean, '.3f'), format(rec_mean, '.3f'), format(pre_mean, '.3f'),
                   format(auc_mean, '.3f'), format(aupr_mean, '.3f'), alltime])
    print(table1)

    time_start2 = time.time()
    Terms = ['MF']
    f_mean, rec_mean, pre_mean, auc_mean, aupr_mean = validation(Terms[0], 5)

    time_end = time.time() - time_start2
    alltime = str(int(time_end) // 60) + 'm' + ' ' + str(int(time_end % 60)) + 's'
    table2 = PrettyTable(['FUNC', 'Fmax_Mean', 'Rec_Mean', 'Prec_Mean', 'AUC_Mean', 'AUPR_Mean', 'AllTime'])
    table2.add_row([Terms[0], format(f_mean, '.3f'), format(rec_mean, '.3f'), format(pre_mean, '.3f'),
                   format(auc_mean, '.3f'), format(aupr_mean, '.3f'), alltime])
    print(table2)

    time_start3 = time.time()
    Terms = ['CC']
    f_mean, rec_mean, pre_mean, auc_mean, aupr_mean = validation(Terms[0], 5)

    time_end = time.time() - time_start3
    alltime = str(int(time_end) // 60) + 'm' + ' ' + str(int(time_end % 60)) + 's'
    table3 = PrettyTable(['FUNC', 'Fmax_Mean', 'Rec_Mean', 'Prec_Mean', 'AUC_Mean', 'AUPR_Mean', 'AllTime'])
    table3.add_row([Terms[0], format(f_mean, '.3f'), format(rec_mean, '.3f'), format(pre_mean, '.3f'),
                   format(auc_mean, '.3f'), format(aupr_mean, '.3f'), alltime])
    print(table3)
    print('Run Over')
    print('*' * 50)
    print(table1)
    print(table2)
    print(table3)
