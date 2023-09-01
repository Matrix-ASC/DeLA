import os
import h5py
from torch.utils.data import Dataset
import torch
from config import data_path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class ModelNet40(Dataset):
    def __init__(self, num_points=1024, partition='train'):
        paths = data_path
        paths = paths.glob(f"ply_data_{partition}*.h5")
        data, label = [], []
        for p in paths:
            f = h5py.File(p, 'r')
            data.append(torch.from_numpy(f['data'][:]).float())
            label.append(torch.from_numpy(f['label'][:]).long())
            f.close()
        self.data = torch.cat(data)
        self.label = torch.cat(label).squeeze()
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, idx):
        pc = self.data[idx][:self.num_points]
        label = self.label[idx]
        if self.partition == 'train':
            scale = torch.rand((3,)) * (3/2 - 2/3) + 2/3
            pc = pc * scale
            pc = pc[torch.randperm(pc.shape[0])]

        return pc*40, label

    def __len__(self):
        return self.data.shape[0]
