import numpy as np 
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA
import torch

def Initialization_D(Z, y_pred, n_clusters, d):
    Z = Z.detach().cpu().numpy()
    D = np.zeros((Z.shape[1], n_clusters * d))
    
    for i in range(n_clusters):
        cluster_mask = (y_pred == i)
        if np.sum(cluster_mask) == 0:
            continue
            
        Z_seperate = Z[cluster_mask]
        n_samples, n_features = Z_seperate.shape
        n_components = min(50, n_samples, n_features)
        
        if n_components < 1:
            continue
            
        pca = PCA(n_components=n_components)
        Z_reduced = pca.fit_transform(Z_seperate.T)
        
        effective_d = min(d, Z_reduced.shape[1])
        u, ss, v = randomized_svd(Z_reduced, n_components=effective_d)
        
        # Modified assignment to handle shape mismatch
        cols_to_assign = min(u.shape[1], d)
        D[:, i*d:i*d+cols_to_assign] = u[:, :cols_to_assign]
    
    return D

#problem s memorijom - prevelika matrica 

# def seperate(Z, y_pred, n_clusters):
#     n, d = Z.shape[0], Z.shape[1]
#     Z_seperate = defaultdict(list)
#     Z_new = np.zeros([n, d])
#     for i in range(n_clusters):
#         for j in range(len(y_pred)):
#             if y_pred[j] == i:
#                 Z_seperate[i].append(Z[j].cpu().detach().numpy())
#                 Z_new[j][:] = Z[j].cpu().detach().numpy()
#     return Z_seperate


# def Initialization_D(Z, y_pred, n_clusters, d):
#     Z_seperate = seperate(Z, y_pred, n_clusters)
#     Z_full = None
#     U = np.zeros([n_clusters * d, n_clusters * d])
#     print("Initialize D")
#     for i in range(n_clusters):
#         Z_seperate[i] = np.array(Z_seperate[i])
#         u, ss, v = np.linalg.svd(Z_seperate[i].transpose())
#         U[:,i*d:(i+1)*d] = u[:,0:d]
#     D = U
#     print("Shape of D: ", D.transpose().shape)
#     print("Initialization of D Finished")
#     return D