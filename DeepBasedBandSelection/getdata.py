# Loads and preprocesses hyperspectral datasets
import numpy as np
import torch
from osgeo import gdal
import scipy.io as sio
from torch.utils.data import Dataset

def get_data(img_path, label_path):
    if img_path[-3:] == 'tif':
        img_data = gdal.Open(img_path).ReadAsArray()
        label_data = gdal.Open(label_path).ReadAsArray()
        img_data = np.transpose(img_data, [1, 2, 0])
        return img_data, label_data
    elif img_path[-3:] == 'mat':
        import scipy.io as sio
        img_mat = sio.loadmat(img_path)
        img_keys = img_mat.keys()
        img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']

        if label_path is not None:
            gt_mat = sio.loadmat(label_path)
            gt_keys = gt_mat.keys()
            gt_key = [k for k in gt_keys if k != '__version__' and k != '__header__' and k != '__globals__']
            return img_mat.get(img_key[0]).astype('float32'), gt_mat.get(gt_key[0]).astype('int8')
        return img_mat.get(img_key[0]).astype('float32'), img_mat.get(img_key[1]).astype('int8')


class Load_my_Dataset(Dataset):
    def __init__(self, data_path, gt_path, patch_size=5):
        super().__init__()
        data_mat = sio.loadmat(data_path)
        print("Available keys in MAT file:", data_mat.keys())
        self.origin_data = data_mat.get('paviaU', data_mat[[k for k in data_mat.keys() if not k.startswith('__')][0]]).astype(np.float32)
        print("Loaded origin_data shape:", self.origin_data.shape)
        self.origin_gt = sio.loadmat(gt_path)[[k for k in sio.loadmat(gt_path).keys() if not k.startswith('__')][0]].copy()
        self.patch_size = patch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self.origin_data.copy()
        self.gt = self.origin_gt.copy()
        self._preprocess_data()
        self.indices = self._generate_valid_indices()
        print(f"Dataset initialized with {len(self.indices)} valid patches")
    
    def _preprocess_data(self):
        margin = self.patch_size // 2
        
        self.data = np.pad(
            self.data,
            pad_width=((margin, margin), (margin, margin), (0, 0)),
            mode='reflect'
        )
        
        self.gt = np.pad(
            self.gt,
            pad_width=((margin, margin), (margin, margin)),
            mode='constant',
            constant_values=0
        )
        
        self.data = (self.data - np.mean(self.data, axis=(0,1))) / np.std(self.data, axis=(0,1))

    def _generate_valid_indices(self):
        margin = self.patch_size // 2
        indices = []
        
        for i in range(margin, self.gt.shape[0] - margin):
            for j in range(margin, self.gt.shape[1] - margin):
                if self.gt[i, j] > 0:
                    indices.append((i, j))
        
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        
        patch = self.data[
            i - self.patch_size//2 : i + self.patch_size//2 + 1,
            j - self.patch_size//2 : j + self.patch_size//2 + 1,
            :
        ]
        
        patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1)  # [C, H, W]
        label = self.gt[i, j] - 1
        
        return patch_tensor, label

    def get_labeled_data(self):
        H, W = self.gt.shape
        B = self.data.shape[2]
        
        data_flat = self.data.reshape(-1, B)
        labels_flat = self.gt.reshape(-1)
        
        mask = labels_flat > 0
        
        return data_flat[mask], labels_flat[mask]
