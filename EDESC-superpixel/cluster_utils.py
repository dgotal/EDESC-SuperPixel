import numpy as np
import torch

def estimate_clusters(similarity_matrix, delta=0.2):
    # similarity_matrix: (bands, bands) numpy array or torch tensor
    if isinstance(similarity_matrix, torch.Tensor):
        W = (torch.abs(similarity_matrix) + torch.abs(similarity_matrix.T)) / 2
        W = W.cpu().numpy()
    else:
        W = (np.abs(similarity_matrix) + np.abs(similarity_matrix.T)) / 2

    D = np.diag(W.sum(axis=1))
    L = D - W
    eigenvalues = np.sort(np.linalg.eigvalsh(L))[::-1]  # descending
    gaps = np.diff(eigenvalues) / eigenvalues[:-1]
    k = np.argmax(gaps >= delta) + 1 if np.any(gaps >= delta) else 1
    num_clusters = len(eigenvalues) - k + 1
    # Add cluster ceiling (max 50% of samples)
    max_clusters = int(0.2 * similarity_matrix.shape[0])  
    return min(num_clusters, max_clusters)
    #return len(eigenvalues) - k + 1
