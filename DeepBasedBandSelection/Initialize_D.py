from collections import defaultdict
import numpy as np
from sklearn.decomposition import TruncatedSVD

def seperate(Z, y_pred, n_clusters):
    """
    Grupira latentne vektore prema klasterima.
    """
    n, d = Z.shape[0], Z.shape[1]
    Z_seperate = defaultdict(list)
    for i in range(n_clusters):
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                Z_seperate[i].append(Z[j])
    return Z_seperate

def Initialization_D(Z, y_pred, n_clusters, d):
    """
    Za svaki klaster radi SVD i uzima d komponenti kao bazu podprostora.
    """
    Z_seperate = seperate(Z, y_pred, n_clusters)
    U = np.zeros([Z.shape[1], n_clusters * d])
    for i in range(n_clusters):
        if len(Z_seperate[i]) == 0:
            continue
        cluster_data = np.array(Z_seperate[i])
        n_components = min(32, cluster_data.shape[1])  # ili n_z
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        try:
            svd.fit(cluster_data)
            u = svd.components_.T
            if u.shape[1] >= d:
                U[:, i*d:(i+1)*d] = u[:, 0:d]
            else:
                U[:, i*d:(i*d)+u.shape[1]] = u[:, :]
        except ValueError:
            print(f"Cluster {i} too small for SVD. Using random initialization.")
            U[:, i*d:(i+1)*d] = np.random.randn(Z.shape[1], d)
    return U
