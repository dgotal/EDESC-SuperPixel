from __future__ import print_function, division
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import cluster_accuracy
import warnings
import torch.optim as optim
import datetime
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from getdata import Load_my_Dataset

try:
    from skfeature.function.similarity_based import reliefF
except ImportError:
    print("scikit-feature nije instaliran. ReliefF nije dostupan.")
    reliefF = None

# I can use BandGate instead of BandAttentionModule
class BandGate(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        self.gates = nn.Parameter(torch.ones(num_bands))

    def forward(self, x):
        mask = torch.sigmoid(self.gates)
        return x * mask.view(1, -1, 1, 1), mask
    
class DownstreamModel(nn.Module):
    def __init__(self, input_dim, num_classes=9):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class BandAttentionModule(nn.Module):
    def __init__(self, num_bands, hidden_dim=64):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(num_bands, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bands)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x).squeeze(-1).squeeze(-1)
        max_pool = self.max_pool(x).squeeze(-1).squeeze(-1)
        attn = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        attn = attn.unsqueeze(-1).unsqueeze(-1)
        return x * attn, attn

class BandSelectionNet(nn.Module):
    def __init__(self, num_bands, n_z, n_clusters):
        super().__init__()
        self.bam = BandAttentionModule(num_bands)
        self.encoder = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, n_z, kernel_size=1),
            nn.BatchNorm2d(n_z)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_z, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_bands, kernel_size=1),
            nn.BatchNorm2d(num_bands)
        )
        self.cluster_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_z, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_clusters)
        )

    def forward(self, x):
        x_attn, attn = self.bam(x)
        z = self.encoder(x_attn)
        x_bar = self.decoder(z)
        clusters = F.softmax(self.cluster_head(z), dim=1)
        return x_bar, clusters, z, attn

def train_EDESC(dataset, device, args, i, epochs=400):
    data, y = dataset.get_labeled_data()
    tensor_data = torch.FloatTensor(data).unsqueeze(-1).unsqueeze(-1).to(device)  # [N, C, 1, 1]
    tensor_labels = torch.LongTensor(y).to(device)
    train_dataset = torch.utils.data.TensorDataset(tensor_data, tensor_labels)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

    class NeuralClustering(nn.Module):
        def __init__(self, n_input, n_z, n_clusters):
            super().__init__()
            self.bam = BandAttentionModule(n_input)
            self.encoder = nn.Sequential(
                nn.Conv2d(n_input, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, n_z, kernel_size=1),
                nn.BatchNorm2d(n_z)
            )
            self.cluster_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_z, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, n_clusters)
            )
            self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_z, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, n_input, kernel_size=1),
            nn.BatchNorm2d(n_input)
            )
        def forward(self, x):
            x, attn = self.bam(x)
            z = self.encoder(x)
            x_bar = self.decoder(z)
            clusters = F.softmax(self.cluster_head(z), dim=1)
            return x_bar, clusters, z, attn

    model = NeuralClustering(args.n_input, args.n_z, args.n_clusters).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )
    
    scaler = torch.cuda.amp.GradScaler()
    best_metrics = {'acc': 0, 'nmi': 0, 'kappa': 0}
    y_np = y 

    for epoch in range(epochs):
        model.train()
        epoch_preds = []
        epoch_labels = []
        
        for batch_idx, (x, labels) in enumerate(train_loader):
            x = x.to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast("cuda"):
                x_bar, clusters, z, attn = model(x)
                recon_loss = F.mse_loss(x_bar, x)
                kl_loss = F.kl_div(clusters.log(), clusters.detach(), reduction='batchmean')
                total_loss = recon_loss + args.alpha * kl_loss

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            epoch_preds.extend(clusters.argmax(1).cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())
        
        acc, kappa, nmi = cluster_accuracy(np.array(epoch_labels), np.array(epoch_preds))
        
        if acc > best_metrics['acc']:
            best_metrics = {'acc': acc, 'nmi': nmi, 'kappa': kappa}
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Acc={acc:.4f}, NMI={nmi:.4f}, Kappa={kappa:.4f}')
            print(f'Loss - Total: {total_loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')    
    
    # Nakon treniranja modela:
    with torch.no_grad():
        x_bar, clusters, z, attn = model(tensor_data)
        attn = attn.mean(dim=0).squeeze()
        selected_band_indices = (attn > 0.6).nonzero().squeeze().cpu().numpy()
        z_flat = z.view(z.size(0), -1).cpu().numpy()

    data_all, labels = dataset.get_labeled_data()
    data_selected = data_all[:, selected_band_indices]

    # SVI BANDOVI
    downstream_all = DownstreamModel(input_dim=data_all.shape[1], num_classes=np.unique(labels).size).to(device)
    preds_all = downstream_all(torch.FloatTensor(data_all).to(device)).argmax(1).cpu().numpy()
    acc_all, kappa_all, nmi_all = cluster_accuracy(labels, preds_all)

    # ODABRANI BANDOVI
    downstream_selected = DownstreamModel(input_dim=data_selected.shape[1], num_classes=np.unique(labels).size).to(device)
    preds_selected = downstream_selected(torch.FloatTensor(data_selected).to(device)).argmax(1).cpu().numpy()
    acc_selected, kappa_selected, nmi_selected = cluster_accuracy(labels, preds_selected)

    print(f"ACC (svi bandovi): {acc_all:.4f}")
    print(f"ACC (odabrani bandovi): {acc_selected:.4f}")

    tsne = TSNE(n_components=2).fit_transform(z_flat)
    plt.scatter(tsne[:,0], tsne[:,1], c=labels, cmap='tab20')
    plt.savefig(f'clusters_{i}.png')

    return best_metrics['acc'], best_metrics['nmi'], best_metrics['kappa'], [], z_flat

def train_band_selection(device, dataset, num_bands, n_z, n_clusters, args, epochs=200):
    data, y = dataset.get_labeled_data()
    tensor_data = torch.FloatTensor(data).unsqueeze(-1).unsqueeze(-1).to(device)
    tensor_labels = torch.LongTensor(y).to(device)
    train_dataset = torch.utils.data.TensorDataset(tensor_data, tensor_labels)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

    model = BandSelectionNet(num_bands, n_z, n_clusters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lambda_sparsity = 0.01

    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, labels) in enumerate(train_loader):
            x = x.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                x_bar, clusters, z, attn = model(x)
                recon_loss = F.mse_loss(x_bar, x)
                kl_loss = F.kl_div(clusters.log(), clusters.detach(), reduction='batchmean')
                sparsity_loss = lambda_sparsity * attn.mean()
                total_loss = recon_loss + args.alpha * kl_loss + sparsity_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={total_loss.item():.4f}")

    with torch.no_grad():
        x_bar, clusters, z, attn = model(tensor_data)
        attn = attn.mean(dim=0).squeeze()
        selected_band_indices = (attn > 0.6).nonzero().squeeze().cpu().numpy()
        print(f"Automatski odabrani bandovi: {selected_band_indices}")
        print(f"Broj odabranih bandova: {len(selected_band_indices)}")

    return model, selected_band_indices, tensor_data, y

def visualize_tsne_with_all_and_selected(tensor_data, selected_band_indices, y):
    # Svi bandovi
    data_all = tensor_data.squeeze(-1).squeeze(-1).cpu().numpy()  # [N, bands]
    tsne_all = TSNE(n_components=2, random_state=42).fit_transform(data_all)
    plt.figure(figsize=(8,6))
    plt.scatter(tsne_all[:,0], tsne_all[:,1], c=y, cmap='tab20', s=10)
    plt.title('t-SNE: Svi bandovi')
    plt.colorbar(label='Class')
    plt.savefig('tsne_all.png')
    plt.close()

    # Odabrani bandovi
    data_selected = tensor_data[:, selected_band_indices, 0, 0].cpu().numpy()
    tsne_selected = TSNE(n_components=2, random_state=42).fit_transform(data_selected)
    plt.figure(figsize=(8,6))
    plt.scatter(tsne_selected[:,0], tsne_selected[:,1], c=y, cmap='tab20', s=10)
    plt.title('t-SNE: Odabrani bandovi')
    plt.colorbar(label='Class')
    plt.savefig('tsne_selected.png')
    plt.close()


def main(args):
    if args.mode == 'band_selection':
        dataset = Load_my_Dataset(
            data_path='datasets/PaviaU.mat',
            gt_path='datasets/PaviaU_gt.mat',
            patch_size=5
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_bands = dataset.origin_data.shape[2]
        model, selected_band_indices, tensor_data, y = train_band_selection(
            device, dataset, num_bands, args.n_z, args.n_clusters, args
        )
        # Svi bandovi
        data_all, labels = dataset.get_labeled_data()  # [N, bands], [N]

        # Odabrani bandovi
        data_selected = data_all[:, selected_band_indices]

        # SVI BANDOVI
        downstream_all = DownstreamModel(input_dim=data_all.shape[1], num_classes=np.unique(labels).size).to(device)
        preds_all = downstream_all(torch.FloatTensor(data_all).to(device)).argmax(1).cpu().numpy()
        acc_all, kappa_all, nmi_all = cluster_accuracy(labels, preds_all)

        # ODABRANI BANDOVI
        downstream_selected = DownstreamModel(input_dim=data_selected.shape[1], num_classes=np.unique(labels).size).to(device)
        preds_selected = downstream_selected(torch.FloatTensor(data_selected).to(device)).argmax(1).cpu().numpy()
        acc_selected, kappa_selected, nmi_selected = cluster_accuracy(labels, preds_selected)

        print(f"ACC (svi bandovi): {acc_all:.4f}")
        print(f"ACC (odabrani bandovi): {acc_selected:.4f}")
    
        visualize_tsne_with_all_and_selected(tensor_data, selected_band_indices, y)

    elif args.mode == 'edesc':
        now = datetime.datetime.now()
        print("hello world " + str(now))

        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))

        device = torch.device("cuda" if args.cuda else "cpu")

        if args.dataset == 'Houston':
            args.pretrain_path = 'weight/Houston.pkl'
            args.n_clusters = 7
            args.n_input = 8
            args.image_size = [130, 130]
            dataset = Load_my_Dataset("/datasets/Houston_corrected.mat", "/datasets/Houston_gt.mat")
            args.num_sample = len(dataset)
        elif args.dataset == 'trento':
            args.pretrain_path = 'weight/trento.pkl'
            args.n_clusters = 6
            args.n_input = 8
            args.image_size = [166, 600]
            dataset = Load_my_Dataset("/datasets/Trento.mat", "/datasets/Trento_gt.mat")
            args.num_sample = len(dataset)
        elif args.dataset == 'pavia':
            args.pretrain_path = ''
            args.n_clusters = 9
            args.image_size = [610, 340]
            dataset = Load_my_Dataset("datasets/PaviaU.mat", "datasets/PaviaU_gt.mat", patch_size=5)
            args.n_input = dataset.data.shape[2]
            print(f"Final input channels: {args.n_input} (should match selected bands)")

        print(args)
        bestacc = bestnmi = best_kappa = acc_sum = nmi_sum = kappa_sum = 0
        rounds = 10
        cas = []
        for i in range(rounds):
            print(f"this is {i} round")
            acc, nmi, kappa, ca, data_reshaped = train_EDESC(dataset, device, args, i)
            acc_sum += acc
            nmi_sum += nmi
            kappa_sum += kappa
            cas.append(ca)
            if acc > bestacc: bestacc = acc
            if nmi > bestnmi: bestnmi = nmi
            if kappa > best_kappa: best_kappa = kappa
        cas = np.array(cas)
        ca = np.mean(cas, axis=0)
        print("cav:", ca)
        print("average_acc:", acc_sum / rounds)
        print("average_nmi:", nmi_sum / rounds)
        print("average_kappa", kappa_sum / rounds)
        print(f'Best ACC {bestacc:.4f}  Best NMI {bestnmi:.4f}  Best kappa {best_kappa:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='band_selection', choices=['band_selection', 'edesc'])
    parser.add_argument('--n_clusters', type=int, default=9)
    parser.add_argument('--n_z', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='pavia')
    parser.add_argument('--alpha', default=3, type=float, help='the weight of kl_loss')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--d', default=5, type=int)
    parser.add_argument('--eta', default=5, type=int)
    parser.add_argument('--pretrain_path', type=str, default='weight/pavia.pkl')
    args = parser.parse_args()
    main(args)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

downstream_model = DownstreamModel(103)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

warnings.filterwarnings("ignore")
epochs = 400

def get_labeled_data(self):
    H, W, B = self.origin_data.shape
    data_reshaped = self.origin_data.reshape(-1, B)
    labels = self.origin_gt.reshape(-1)
    mask = labels > 0
    print(f"Labeled data shape: {data_reshaped[mask].shape}")
    return data_reshaped[mask], labels[mask]
