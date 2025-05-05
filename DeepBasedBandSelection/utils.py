from __future__ import division, print_function
import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score
from sklearn import metrics
import torch

def cluster_accuracy(y_true, y_pre, return_aligned=False):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy().astype('float32')
    if isinstance(y_pre, torch.Tensor):
        y_pre = y_pre.cpu().numpy().astype('float32')
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]

    # Calculate accuracy
    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    acc = 1. - missrate
    nmi = metrics.normalized_mutual_info_score(y_true, y_pre)
    kappa = metrics.cohen_kappa_score(y_true, y_best)
    ari = metrics.adjusted_rand_score(y_true, y_best)
    fscore = metrics.f1_score(y_true, y_best, average='micro')
    ca = class_acc(y_true, y_best)
    if return_aligned:
        return y_best, acc, kappa, nmi, ca  # 5 values
    else:
        return acc, kappa, nmi  # 3 values


def class_acc(y_true, y_pre):
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
        y_c_p = y_pre[np.nonzero(y_true == c)]
        acurracy = accuracy_score(y_c, y_c_p)
        ca.append(acurracy)
    ca = np.array(ca)
    return ca
