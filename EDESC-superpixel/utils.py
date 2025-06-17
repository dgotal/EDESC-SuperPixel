from __future__ import division, print_function
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    # Get indices as paired tuples
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum(w[row_ind[i], col_ind[i]] for i in range(len(row_ind))) / y_pred.size

