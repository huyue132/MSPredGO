import time, torch, warnings
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from msdata import read_benchmark, WeightData
from evaluate import cacul_aupr, calculate_performance, train_step
from utils import Thresholds
from config import get_config
from model import  Seq_Module, Domain_Module, PPI_Module, Weight_classifier

args = get_config()
warnings.filterwarnings('ignore')


def train(train_benchmark, test_benchmark, func, device= args.device):
    learners = ['sequence', 'domain', 'ppi']
    seq_learner = Seq_Module(func).to(device)
    domain_learner = Domain_Module(func).to(device)
    ppi_learner = PPI_Module(func).to(device)
    if func == 'BP':
        seq_epoch = 1  # 30
        seq_lr = 0.0001
        seq_batch_size = 16
        domain_epoch = 1   # 45
        domain_lr = 0.001
        domain_batch_size = 32
    else:
        seq_epoch = 1  # 35
        seq_lr = 0.001
        seq_batch_size = 8
        domain_epoch = 1   # 35
        domain_lr = 0.001
        domain_batch_size = 16
    ppi_epoch = 1  # 40
    ppi_lr = 0.0001
    ppi_batch_size = 8
    for learner in learners:
        if learner == 'sequence':
            seq_train_out, seq_test_out, seq_th = train_step(model=seq_learner,
                                                             batch_size=seq_batch_size,lr=seq_lr,
                                                             learner=learner, device=device,
                                                             epochs=seq_epoch, func=func,
                                                             train_benchmark=train_benchmark,
                                                             test_benchmark=test_benchmark) # 30
        elif learner == 'domain':
            domain_train_out, domain_test_out, domain_th = train_step(model=domain_learner,
                                                             batch_size=domain_batch_size,lr=domain_lr,
                                                             learner=learner, device=device,
                                                             epochs=domain_epoch, func=func,
                                                             train_benchmark=train_benchmark,
                                                             test_benchmark=test_benchmark) # 45
        elif learner == 'ppi':
                ppi_train_out, ppi_test_out, ppi_th = train_step(model=ppi_learner,
                                                                 batch_size=ppi_batch_size,lr=ppi_lr,
                                                                 learner=learner, device=device,
                                                                 epochs=ppi_epoch, func=func,
                                                                 train_benchmark=train_benchmark,
                                                                 test_benchmark=test_benchmark) # 40
    print('{}  Weight_model start'.format(func))
    learning_rate = 0.001
    batch_size = 32
    epoch_times = 1    # 30
    weight_model = Weight_classifier(func).to(device)
    loss_function = nn.BCELoss()
    optimizer = Adam(weight_model.parameters(), lr=learning_rate, weight_decay=0.00001)
    train_dataset = WeightData(train_benchmark, seq_train_out, domain_train_out, ppi_train_out, func=func)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = WeightData(test_benchmark, seq_test_out, domain_test_out, ppi_test_out, func=func)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    print('\nWeight training begins')
    print("{:^7} {:^10} {:^10} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8}".format('Epoch', 'Loss', 'TestLoss', 'T',
                                                                                 'F_score',
                                                                                 'AUC', 'AUPR', 'Recall', 'Prec',
                                                                                 'EpochTime'))
    print('-' * 95)
    wsince = time.time()
    best_fscore = 0
    for epoch in range(epoch_times):
        _loss = 0
        batch_num = 0
        weight_model.train()
        for batch_idx, (weight_features, label) in enumerate(train_data_loader):
            since = time.time()
            weight_features = Variable(weight_features).to(device)
            GO_annotiations = torch.squeeze(label)
            GO_annotiations = Variable(GO_annotiations).to(device)
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
        weight_model.eval()
        for idx, (weight_features, labels) in enumerate(test_data_loader):
            weight_features = Variable(weight_features).to(device)
            labels = Variable(labels).to(device)
            out = weight_model(weight_features)
            test_batch_num = test_batch_num + 1
            pred.append(out.data[0].cpu().tolist())
            actual.append(labels.data[0].cpu().tolist())
            one_loss = loss_function(out, labels)
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
    w_elapsed = time.time() - wsince
    test_loss = float(test_loss)
    f_max = float(f_max)
    bestauc_score = float(bestauc_score)
    bestaupr_score = float(aupr_score)
    recall_max = float(recall_max)
    prec_max = float(prec_max)
    bestthreshold = float(bestthreshold)
    table = PrettyTable(['TestLoss', 'Lr', 'Batch', 'Epoch', 'Fmax', 'AUC', 'AUPR', 'Recall', 'Prec', 'T', 'Time', ])
    modeltime = str(int(w_elapsed) // 60) + 'm' + ' ' + str(int(w_elapsed % 60)) + 's'
    table.add_row([format(test_loss, '.5f'), learning_rate, batch_size, epoch_times, format(f_max, '.3f'),
                   format(bestauc_score, '.3f'), format(bestaupr_score, '.3f'), format(recall_max, '.3f'),
                   format(prec_max, '.3f'), format(bestthreshold, '.3f'), modeltime])
    print(table)
    test_weight_model = torch.load(
        'savedpkl/WeightVal_{}_{}_{}_{}.pkl'.format(func, batch_size, learning_rate, epoch_times)).to(device)
    t_loss = 0
    weight_test_outs = {}
    pred = []
    actual = []
    score_dict = {}
    batch_num = 0
    for batch_idx, (weight_features, labels) in enumerate(test_data_loader):
        weight_features = Variable(weight_features).to(device)
        labels = Variable(labels).to(device)
        out = test_weight_model(weight_features)
        batch_num += 1
        weight_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
        pred.append(out.data[0].cpu().tolist())
        actual.append(labels.data[0].cpu().tolist())
        loss = loss_function(out, labels)
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
    bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    prec_max, bestauc_score, bestaupr_score = each_best_scores[3], each_best_scores[4], each_best_scores[5]
    with open('/home/new2/huyuecode/sdn2go/out/weight_out/Weightout{}_lr{}_bat{}_epo{}.csv'.format(
            func, learning_rate, batch_size, epoch_times), 'w') as f:
        f.write('lr:{},batchsize:{},epochtimes:{}\n'.format(learning_rate, batch_size, epoch_times))
        f.write('f_max:{},recall_max{},prec_max{},auc_score:{}, aupr:{}\n'.format(
            f_max, recall_max, prec_max, bestauc_score, bestaupr_score))
        f.write('threshold,f_score,recall,precision,auc,aupr\n')
        for i in range(len(Thresholds)):
            f.write('{},'.format(str(Thresholds[i])))
            f.write('{}\n'.format(','.join(str(x) for x in score_dict[Thresholds[i]])))
        for key, var in weight_test_outs.items():
            f.write('{},'.format(str(key)))
            f.write('{}\n'.format(','.join(str(x) for x in var)))
    return each_best_scores

def validation(func='MF', k_fold=5):
    kf = KFold(n_splits=k_fold)
    benchmark = np.array(read_benchmark())
    scores = []
    for train_index, test_index in kf.split(benchmark):
        train_set = benchmark[train_index].tolist()
        test_set = benchmark[test_index].tolist()
        each_fold_scores = train(train_benchmark=train_set, test_benchmark=test_set, func=func)
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
    f_mean, rec_mean, pre_mean, auc_mean, aupr_mean = validation(args.onto, 5)
    time_end = time.time() - time_start1
    alltime = str(int(time_end) // 60) + 'm' + ' ' + str(int(time_end % 60)) + 's'
    table1 = PrettyTable(['FUNC', 'Fmax_Mean', 'Rec_Mean', 'Prec_Mean', 'AUC_Mean', 'AUPR_Mean', 'AllTime'])
    table1.add_row([args.onto, format(f_mean, '.3f'), format(rec_mean, '.3f'), format(pre_mean, '.3f'),
               format(auc_mean, '.3f'), format(aupr_mean, '.3f'), alltime])
    print(table1)
