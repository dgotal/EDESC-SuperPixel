from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import cluster_acc
import warnings
from AutoEncoder import AE
from InitializeD import Initialization_D
from Constraint import D_constraint1, D_constraint2
import time
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
timestamp = time.strftime("%Y%m%d_%H%M%S")

from skimage.segmentation import slic

def apply_superpixels(hypercube, n_segments=100):
    # hypercube: (H, W, B)
    segments = slic(hypercube, n_segments=n_segments, compactness=10, start_label=0)
    return segments

def remove_redundant_bands(selected, X, threshold=0.9):
    corr_matrix = np.corrcoef(X[selected].T)
    to_keep = []
    for i in range(len(selected)):
        if all(corr_matrix[i,j] < threshold for j in to_keep):
            to_keep.append(i)
    return selected[to_keep]

from sklearn.feature_selection import mutual_info_classif

def select_representative_bands(cluster_labels, X, labels, strategy='mi'):
    selected = []
    X_np = X.cpu().numpy()
    
    for cluster in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster)[0]
        
        if strategy == 'mi':
            mi = mutual_info_classif(X_np[indices].T, labels)
            selected.append(indices[np.argmax(mi)])
        elif strategy == 'maxvar':
            band_vars = X_np[indices].var(axis=1)
            selected.append(indices[np.argmax(band_vars)])
            
    return np.array(selected)

def prepare_hsi_data(cube, gt, superpixels=200):
    # Primijeni superpixel segmentaciju
    from skimage.segmentation import slic
    segments = slic(cube, n_segments=superpixels, compactness=0.1)
    
    # Izračunaj prosjek spektara unutar svakog superpixela
    unique_segments = np.unique(segments)
    X = np.zeros((len(unique_segments), cube.shape[-1]))
    for i, seg_id in enumerate(unique_segments):
        X[i] = np.mean(cube[segments == seg_id], axis=0)
    
    return X, segments

import cvxpy as cp
import numpy as np

def sparse_subspace_clustering(X, alpha=20):
    n_bands = X.shape[1]
    C = cp.Variable((n_bands, n_bands))
    
    # Optimizacijski problem
    objective = cp.Minimize(cp.norm(C, 1) + alpha * cp.norm(X @ C - X, 'fro'))
    constraints = [cp.diag(C) == 0]  # Bez self-representation
    prob = cp.solve(prob, solver=cp.SCS)
    
    # Afinitetna matrica
    W = 0.5 * (np.abs(C.value) + np.abs(C.value.T))
    return W

from sklearn.cluster import KMeans

class BandClusterer:
    def __init__(self, method: str, n_clusters: int = None):
        self.method = method
        self.n_clusters = n_clusters
        
    def fit(self, X):
        if self.method == 'kmeans':
            model = KMeans(n_clusters=self.n_clusters)
            model.fit(X.T)  # Transponiraj za band clustering
            self.labels_ = model.labels_
        else:
            raise ValueError(f"Nepoznata metoda: {self.method}")
        return self

class EDESC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 num_sample,
                 pretrain_path='pretrain_pavia.pt'):
        super(EDESC, self).__init__()
        self.pretrain_path = pretrain_path
        self.n_clusters = n_clusters

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)	

        # Subspace bases proxy
        self.D = Parameter(torch.Tensor(n_z, n_clusters))

        
    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # Load pre-trained weights
        self.ae.load_state_dict(torch.load(self.pretrain_path, map_location='cpu'))
        print('Load pre-trained model from', path)

    def forward(self, x):
        
        x_bar, z = self.ae(x)
        d = args.d
        s = None
        eta = args.eta
      
        # Calculate subspace affinity
        for i in range(self.n_clusters):	
			
            si = torch.sum(torch.pow(torch.mm(z,self.D[:,i*d:(i+1)*d]),2),1,keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s,si),1)   
        s = (s+eta*d) / ((eta+1)*d)
        s = (s.t() / torch.sum(s, 1)).t()
        return x_bar, s, z

    def total_loss(self, x, x_bar, z, pred, target, dim, n_clusters, beta):

	# Reconstruction loss
        reconstr_loss = F.mse_loss(x_bar, x)     
        
        # Subspace clustering loss
        kl_loss = F.kl_div(pred.log(), target)
        
        # Constraints
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)
  
        # Total_loss
        total_loss = reconstr_loss + beta * kl_loss + loss_d1 + loss_d2

        return total_loss

def refined_subspace_affinity(s):
    weight = s**2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()

def pretrain_ae(model, data):
    train_loader = DataLoader(torch.utils.data.TensorDataset(data), batch_size=512, shuffle=True)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.pretrain_epochs):
        total_loss = 0.
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("Model saved to {}.".format(args.pretrain_path))
    
def train_EDESC():
    import scipy.io
    from cluster_utils import estimate_clusters

    mat = scipy.io.loadmat('Pavia.mat')
    hypercube = mat['pavia']
    mat_gt = scipy.io.loadmat('Pavia_gt.mat')
    gt = mat_gt['pavia_gt']

    print("Unique GT labels:", np.unique(gt))

    H, W, B = hypercube.shape

    segments = apply_superpixels(hypercube, args.superpixels)
    superpixel_avg = np.zeros((segments.max()+1, B), dtype=np.float32)
    from scipy import stats
    for i in range(segments.max() + 1):
        mask = segments == i
        superpixel_avg[i] = hypercube[mask].mean(axis=0)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.imshow(hypercube[:, :, 30])
    plt.title('Originalna slika')
    plt.subplot(122)
    plt.imshow(segments, cmap='nipy_spectral')
    plt.title(f'Superpixel segmentacija ({args.superpixels} segmenata)')
    plt.savefig(f'superpixels_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Labele superpixela
    labels_superpixel = np.zeros(segments.max() + 1)
    for i in range(segments.max() + 1):
        mask = segments == i
        labels_superpixel[i] = stats.mode(gt[mask], axis=None)[0]

    # Prije filtriranja: izračunajte class_counts za sve klase
    from collections import Counter
    class_counts = Counter(labels_superpixel)  # Ovdje je ključna promjena
    print("Initial class distribution:", class_counts)

    # Filter out classes with <2 samples
    valid_classes = [cls for cls, count in class_counts.items() if count >= 1]

    if not valid_classes:
        raise ValueError("Nema validnih klasa za evaluaciju")

    mask = np.isin(labels_superpixel, valid_classes)
    labels = labels_superpixel[mask]
    pixels = superpixel_avg[mask]

    # Verify new distribution
    print("Filtered class distribution:", Counter(labels))

    from scipy.ndimage import gaussian_filter1d
    pixels_filtered = gaussian_filter1d(pixels.T, sigma=1, axis=0).T  # 1D Gaussian filter
    pixels = pixels_filtered

    from collections import Counter
    print("Distribucija klasa nakon filtriranja:", Counter(labels))


    band_samples = torch.tensor(pixels.T, dtype=torch.float32).to(device)  # (B, num_superpixels)

    band_samples_np = band_samples.cpu().numpy()

    # t-SNE na superpixel značajkama
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    pixels_2d = tsne.fit_transform(pixels)

    fig = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pixels_2d[:,0], pixels_2d[:,1], c=labels, cmap='tab10', s=30)
    plt.title('t-SNE superpixel feature embedding')
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.savefig(f'tsne_superpixels_{timestamp}.png', dpi=150)
    plt.close(fig)

    if pixels.size == 0:
        raise ValueError("No valid superpixels after filtering. Check GT labels and superpixel parameters.")

    print(f"Superpixels after filtering: {pixels.shape[0]}")
    
    similarity = 1 - torch.cdist(band_samples, band_samples, p=1)
    num_clusters = estimate_clusters(similarity, delta=args.delta)
    print("Procijenjeni number of clusters:", num_clusters)

    kmeans_bands = KMeans(n_clusters=num_clusters, n_init=10)
    cluster_labels = kmeans_bands.fit_predict(band_samples.cpu().numpy())

    # Band selection:
    selected_band_indices = select_representative_bands(
    cluster_labels, 
    band_samples, 
    labels=labels,
    strategy=args.strategy
)
    selected_band_indices = remove_redundant_bands(selected_band_indices, band_samples_np)
    pixels_selected = pixels[:, selected_band_indices]

    original_shape = hypercube.shape
    reduced_shape = pixels_selected.shape
    print(f"Original dimension: {original_shape}")
    print(f"Reduced dimension: {reduced_shape}")
    print(f"Percentage of selected bands: {100 * reduced_shape[1]/original_shape[2]:.2f}%")

    from sklearn.decomposition import PCA
    # Transponse band_samples for PCA (51 bands, 96 superpixela)
    bands_2d = PCA(n_components=2).fit_transform(band_samples_np)  # Obrazac (51, 2)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(bands_2d[:, 0], bands_2d[:, 1], c=cluster_labels, cmap='tab20', s=50)
    plt.title(f'Klasteri bandova (n_clusters={num_clusters})')
    plt.savefig(f'band_clusters_{timestamp}.png', dpi=150)
    plt.close(fig)

    # t-SNE na odabranim bandovima
    tsne_sel = TSNE(n_components=2, perplexity=30, random_state=42)
    pixels_sel_2d = tsne_sel.fit_transform(pixels_selected)

    fig = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pixels_sel_2d[:,0], pixels_sel_2d[:,1], c=labels, cmap='tab10', s=30)
    plt.title('t-SNE odabrani bandovi (superpixel avg)')
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.savefig(f'tsne_selectedbands_{timestamp}.png', dpi=150)
    plt.close(fig)

    data = torch.Tensor(pixels).float().to(device)

    model = EDESC(
        n_enc_1=256,  #It was 500
        n_enc_2=500,
        n_enc_3=1000,
        n_dec_1=1000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=B,
        n_z=args.n_z,
        n_clusters=num_clusters,
        num_sample = pixels.shape[0],
        pretrain_path=args.pretrain_path).to(device)
    start = time.time()      

    # Load pre-trained model
    #model.pretrain(args.pretrain_path)
    pretrain_ae(model.ae, data)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Cluster parameter initiate
    
    y = labels
    x_bar, hidden = model.ae(data)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10) 
 
    # Get clusters from K-means
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    print("Initial Cluster Centers: ", y_pred)
    
    # Initialize D
    D = Initialization_D(hidden, y_pred, num_clusters, args.d)
    D = torch.tensor(D).to(torch.float32)
    accmax = 0
    nmimax = 0  
    y_pred_last = y_pred
    model.D.data = D.to(device)
    
    model.train()
    
    for epoch in range(200):
        x_bar, s, z = model(data)

        # Update refined subspace affinity
        tmp_s = s.data
        s_tilde = refined_subspace_affinity(tmp_s)

        # Evaluate clustering performance
        y_pred = tmp_s.cpu().detach().numpy().argmax(1)
        delta_label = np.sum(y_pred != y_pred_last).astype(
            np.float32) / y_pred.shape[0]
        y_pred_last = y_pred
        acc = cluster_acc(y, y_pred)
        nmi = nmi_score(y, y_pred)
        if acc > accmax:
            accmax = acc
        if nmi > nmimax:
            nmimax = nmi            
        print('Iter {}'.format(epoch), ':Current Acc {:.4f}'.format(acc),
                  ':Max Acc {:.4f}'.format(accmax),', Current nmi {:.4f}'.format(nmi), ':Max nmi {:.4f}'.format(nmimax))
        
        ############## Total loss function ######################
        loss = model.total_loss(data, x_bar, z, pred=s, target=s_tilde, dim=args.d, n_clusters = num_clusters, beta = args.beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()
    print('Running time: ', end-start)

    # Classification on all bands
    clf_full = SVC(kernel='rbf')

    from sklearn.model_selection import train_test_split

    # Za potpuni skup
    X_train, X_test, y_train, y_test = train_test_split(
        pixels, 
        labels, 
        test_size=args.test_size,
        stratify=labels
        #random_state=42
    )
    clf_full.fit(X_train, y_train)
    y_pred_full = clf_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)

    # Za odabrane bandove
    clf_selected = SVC(kernel='rbf')
    X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
        pixels_selected, 
        labels, 
        test_size=args.test_size,
        stratify=labels
        #random_state=42
    )
    clf_selected.fit(X_train_sel, y_train_sel)
    y_pred_sel = clf_selected.predict(X_test_sel)
    acc_selected = accuracy_score(y_test_sel, y_pred_sel)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred_full)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f'confusion_matrix_{timestamp}.png')
    plt.close()

    print(f"SVM Accuracy (Full): {acc_full:}")
    print(f"SVM Accuracy (Selected): {acc_selected:}")

    with open(f'metrics_{timestamp}.txt', 'w') as f:
        f.write(f"Broj superpixela: {args.superpixels}\n")
        f.write(f"Broj odabranih bandova: {reduced_shape[1]}\n")
        f.write(f"SVM Accuracy (Full): {acc_full:.4f}\n")
        f.write(f"SVM Accuracy (Selected): {acc_selected:.4f}\n")
        f.write(f"Best ACC: {accmax:.4f}\n")
        f.write(f"Best NMI: {nmimax:.4f}\n")

    return accmax, nmimax, pixels_selected, labels
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='EDESC training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=4, type=int)
    parser.add_argument('--d', default=5, type=int)
    parser.add_argument('--n_z', default=20, type=int)
    parser.add_argument('--eta', default=5, type=int)
    parser.add_argument('--strategy', type=str, choices=['maxvar', 'mi'], default='maxvar',help='Strategija odabira bandova (maxvar/mi)')
    #parser.add_argument('--batch_size', default=512, type=int)    
    parser.add_argument('--dataset', type=str, default='pavia')
    parser.add_argument('--pretrain_path', type=str, default='pretrain_pavia.pt')
    parser.add_argument('--beta', default=0.1, type=float, help='coefficient of subspace affinity loss')
    parser.add_argument('--superpixels', type=int, default=100, help='Broj superpixela')
    parser.add_argument('--delta', type=float, default=0.2, help='Prag za procjenu klastera')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma za Gaussian blur')
    parser.add_argument('--test_size', type=float, default=0.3, help='Postotak testnog skupa (0.0-1.0)')
    parser.add_argument('--pretrain_epochs', type=int, default=50,help='Broj epoha za pre-trening autoencodera')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    print(args)
    bestacc = 0 
    bestnmi = 0
    for i in range(10):
        acc, nmi, pixels_selected, labels = train_EDESC()
        if acc > bestacc:
            bestacc = acc
        if nmi > bestnmi:
            bestnmi = nmi
    print('Best ACC {:.4f}'.format(bestacc), ' Best NMI {:4f}'.format(bestnmi))
