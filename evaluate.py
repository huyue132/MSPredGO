import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics
import time, torch, os
from utils import Thresholds, OUT_nodes
from torch.nn import BCELoss
from prettytable import PrettyTable
from msdata import MSData

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


def train_step(model, train_benchmark, test_benchmark, lr, learner, device, epochs, func, batch_size):
    print(model)
    train_dataset = MSData(train_benchmark, func)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MSData(test_benchmark, func)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f'{learner} learner starting')
    loss_function = BCELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.00001)
    save_path = '/home/new2/huyuecode/sdn2go/save_model'
    since = time.time()
    print(
        "{:^7} {:^10} {:^10} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8}".format('Epoch', 'Loss', 'TestLoss', 'Lr',
                                                                                     'T', 'F_score', 'AUC', 'AUPR',
                                                                                     'Recall', 'Prec', 'EpochTime'))
    print('-' * 103)
    best_fscore = 0
    for epoch in range(epochs):
        _loss = 0
        batch_num = 0
        since = time.time()
        lrs = optimizer.param_groups[0]['lr']
        model.train()
        for batch_idx, (seq_feat, domain_feat, ppi_feat, labels) in enumerate(train_loader):
            if learner == 'sequence':
                feat = Variable(seq_feat).to(device)
            elif learner == 'domain':
                feat = Variable(domain_feat).to(device)
            elif learner == 'ppi':
                feat = Variable(ppi_feat)
            labels = Variable(labels).to(device)
            out = model(feat)
            optimizer.zero_grad()
            loss = loss_function(out, labels)
            batch_num += 1
            loss.backward()
            optimizer.step()
            _loss += loss.item()
        epoch_loss = "{}".format(_loss / batch_num)
        t_loss = 0
        test_batch_num = 0
        pred = []
        actual = []

        model.eval()  # 开启评估模式
        for idx, (seq_feat, domain_feat, ppi_feat, labels) in enumerate(test_loader):
            if learner == 'sequence':
                feat = Variable(seq_feat).cuda()
            elif learner == 'domain':
                feat = Variable(domain_feat).cuda()
            elif learner == 'ppi':
                feat = Variable(ppi_feat).cuda()
            labels = Variable(labels).cuda()
            out = model(feat)
            test_batch_num = test_batch_num + 1
            pred.append(out.data[0].cpu().tolist())
            actual.append(labels.data[0].cpu().tolist())
            labels = labels.squeeze(-1)
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
            torch.save(model,os.path.join(save_path,f'{learner}_{func}_{batch_size}_{lr}_{epochs}.pkl'))
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
              format(format(lrs, '.5f'), '^8'),
              format(format(t, '.3f'), '^8'),
              format(format(f_score, '.3f'), '^8'),
              format(format(auc_score, '.3f'), '^8'),
              format(format(aupr_score, '.3f'), '^8'),
              format(format(recall, '.3f'), '^8'),
              format(format(precision, '.3f'), '^8'),
              format(epochtime, '^8')
              )
        # scheduler.step()
    bestthreshold, f_max, recall_max = best_scores[0], best_scores[1], best_scores[2]
    prec_max, bestauc_score, bestaupr_score = best_scores[3], best_scores[4], best_scores[5]
    seq_elapsed = time.time() - since
    f_max = float(f_max)
    bestauc_score = float(bestauc_score)
    bestaupr_score = float(bestaupr_score)
    recall_max = float(recall_max)
    prec_max = float(prec_max)
    bestthreshold = float(bestthreshold)
    print('Model training ends')
    print('\nModel validation begins')
    table = PrettyTable(['TestLoss', 'Lr', 'Batch', 'Epoch', 'Fmax', 'AUC', 'AUPR', 'Recall', 'Prec', 'T', 'Time', ])
    modeltime = str(int(seq_elapsed) // 60) + 'm' + ' ' + str(int(seq_elapsed % 60)) + 's'
    table.add_row([format(test_loss, '.5f'), lr, batch_size, epochs, format(f_max, '.3f'),
                   format(bestauc_score, '.3f'), format(bestaupr_score, '.3f'), format(recall_max, '.3f'),
                   format(prec_max, '.3f'), format(bestthreshold, '.3f'), modeltime])
    print(table)
    print('Model validation ends')
    # return f_max, bestauc_score, bestaupr_score, recall_max, prec_max, seq_elapsed
    test_model = torch.load(os.path.join(save_path,f'{learner}_{func}_{batch_size}_{lr}_{epochs}.pkl')).to(device)

    t_loss = 0
    seq_test_outs = {}
    pred = []
    actual = []
    score_dict = {}
    batch_num = 0
    print('Model test start')
    for batch_idx, (seq_feat, domain_feat, ppi_feat, labels) in enumerate(test_loader):
        if learner == 'sequence':
            feat = Variable(seq_feat).cuda()
        elif learner == 'domain':
            feat = Variable(domain_feat).cuda()
        elif learner == 'ppi':
            feat = Variable(ppi_feat).cuda()
        out = test_model(feat)
        out = out.squeeze(-1)
        batch_num += 1
        seq_test_outs[test_benchmark[batch_idx]] = out.data[0].cpu().tolist()
        pred.append(out.data[0].cpu().tolist())
        actual.append(labels.data[0].cpu().tolist())
        loss = loss_function(out.cpu(), labels)
        t_loss += loss.item()
    test_loss = "{}".format(t_loss / batch_num)
    fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    aupr_score = cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())
    each_best_fcore = 0
    for i in range(len(Thresholds)):
        f_score, recall, precision = calculate_performance(
            actual, pred, threshold=Thresholds[i], average='micro')
        if f_score > each_best_fcore:
            each_best_fcore = f_score
            each_best_scores = [Thresholds[i], f_score, recall, precision, auc_score, aupr_score]
        scores = [f_score, recall, precision, auc_score, aupr_score]
        score_dict[Thresholds[i]] = scores
    bestthreshold, f_max, recall_max = each_best_scores[0], each_best_scores[1], each_best_scores[2]
    prec_max, bestauc_score, bestaupr_score = each_best_scores[3], each_best_scores[4], each_best_scores[5]
    test_loss = float(test_loss)
    f_max = float(f_max)
    bestthreshold = float(bestthreshold)
    auc_score = float(auc_score)
    recall_max = float(recall_max)
    prec_max = float(prec_max)
    print('Model test ends')
    weight_path = '/home/new2/huyuecode/sdn2go/out/weight_out'
    with open(os.path.join(weight_path,f'{learner}_lr{lr}_bat{batch_size}_epo{epochs}.csv'), 'w') as f:
        f.write(f'lr:{lr},batchsize:{batch_size},epochtimes:{epochs}\n')
        f.write('f_max:{},recall_max{},prec_max{},auc_score:{},aupr_score:{}\n'.format(
            f_max, recall_max, prec_max, auc_score, aupr_score))
        f.write('threshold,f_score,recall,precision,auc,aupr\n')
        for i in range(len(Thresholds)):
            f.write('{},'.format(str(Thresholds[i])))
            f.write('{}\n'.format(','.join(str(x) for x in score_dict[Thresholds[i]])))
        for key, var in seq_test_outs.items():
            f.write('{},'.format(str(key)))
            f.write('{}\n'.format(','.join(str(x) for x in var)))

    # 获取再最优模型下的训练集的输出
    train_out_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    seq_train_outs = {}
    for batch_idx, (seq_feat, domain_feat, ppi_feat, labels) in enumerate(train_out_loader):
        if learner == 'sequence':
            feat = Variable(seq_feat).cuda()
        elif learner == 'domain':
            feat = Variable(domain_feat).cuda()
        elif learner == 'ppi':
            feat = Variable(ppi_feat).cuda()
        out = model(feat)
        out = out.squeeze(-1)
        seq_train_outs[train_benchmark[batch_idx]] = out.data[0].cpu().tolist()
    return seq_train_outs, seq_test_outs, bestthreshold
