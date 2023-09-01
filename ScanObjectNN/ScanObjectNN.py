import os
import h5py
import random, math
import torch
from torch.utils.data import Dataset
from config import data_path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class ScanObjectNN(Dataset):
    def __init__(self, partition='training'):
        h5_name = data_path / f"main_split/{partition}_objectdataset_augmentedrot_scale75.h5"
        f = h5py.File(h5_name, mode="r")
        self.data = torch.from_numpy(f['data'][:]).float()
        self.label = torch.from_numpy(f['label'][:]).type(torch.uint8)
        f.close()
        self.partition = partition

    def __getitem__(self, idx):
        pc = self.data[idx]
        label = self.label[idx]
        if self.partition == 'training':
            angle = random.random() * 2 * math.pi
            cos, sin = math.cos(angle), math.sin(angle)
            rotmat = torch.tensor([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
            scale = torch.rand((3,)) * 0.2 + 0.9
            rotmat = torch.diag(scale) @ rotmat
            pc = pc @ rotmat
            pc = pc[torch.randperm(pc.shape[0])]

        return pc.mul(40), label

    def __len__(self):
        return self.data.shape[0]
