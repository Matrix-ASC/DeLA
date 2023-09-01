import torch, numpy as np
from torch.nn import functional as F
import random
import math
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.cutils import grid_subsampling, KDTree, grid_subsampling_test
from config import processed_data_path

class S3DIS(Dataset):
    r"""
    partition   =>   areas, can be "2"  "23"  "!23"==="1456"   
    train=True  =>   training   
    train=False =>   validating 
    test=True   =>   testing    
    warmup=True =>   warmup     

    args: 
    k           =>   k in knn, [k1, k2, ..., kn]   
    grid_size   =>   as in subsampling, [0.04, 0.06, ..., 0.3]   
                        if warmup is True, should be estimated (lower) downsampling ratio except the first: [0.04, 2, ..., 2.5]  
    max_pts     =>   optional, max points per sample when training  
    """
    def __init__(self, args, partition="!5", loop=30, train=True, test=False, warmup=False):

        self.paths = list(processed_data_path.glob(f'[{partition}]*.pt'))

        self.loop = loop
        self.train = train
        self.test = test
        self.warmup = warmup

        self.k = list(args.k)
        self.grid_size = list(args.grid_size)
        self.max_pts = 2**3**4
        if hasattr(args, "max_pts") and args.max_pts > 0:
            self.max_pts = args.max_pts
        
        if warmup:
            maxpts = 0
            for p in self.paths:
                s = torch.load(p)[0].shape[0]
                if s > maxpts:
                    maxpts = s
                    maxp = p
            self.paths = [maxp]
        
        self.datas = [torch.load(path) for path in self.paths]


    def __len__(self):
        return len(self.paths) * self.loop
    
    def __getitem__(self, idx):
        if self.test:
            return self.get_test_item(idx)

        idx //= self.loop
        xyz, col, lbl = self.datas[idx]

        if self.train:
            angle = random.random() * 2 * math.pi
            cos, sin = math.cos(angle), math.sin(angle)
            rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
            rotmat *= random.uniform(0.8, 1.2)
            xyz = xyz @ rotmat
            xyz += torch.empty_like(xyz).normal_(std=0.005)
            xyz -= xyz.min(dim=0)[0]

        # here grid size is assumed 0.04, so estimated downsampling ratio is ~14
        if self.train:
            indices = grid_subsampling(xyz, self.grid_size[0], 2.5 / 14)
        else:
            indices = grid_subsampling_test(xyz, self.grid_size[0], 2.5 / 14, pick=0)

        xyz = xyz[indices]

        if not self.train:
            xyz -= xyz.min(dim=0)[0]

        if xyz.shape[0] > self.max_pts and self.train:
            pt = random.choice(xyz)
            condition = (xyz - pt).square().sum(dim=1).argsort()[:self.max_pts].sort()[0]  # sort to preserve locality
            xyz = xyz[condition]
            indices = indices[condition]
        
        col = col[indices]
        lbl = lbl[indices]
        col = col.float()

        if self.train and random.random() < 0.2:
            col.fill_(0.)
        else:
            if self.train and random.random() < 0.2:
                colmin = col.min(dim=0, keepdim=True)[0]
                colmax = col.max(dim=0, keepdim=True)[0]
                scale = 255 / (colmax - colmin)
                alpha = random.random()
                col = (1 - alpha + alpha * scale) * col - alpha * colmin * scale
            col.mul_(1 / 250.)
        
        height = xyz[:, 2:]
        feature = torch.cat([col, height], dim=1)

        indices = []
        self.knn(xyz, self.grid_size[::-1], self.k[::-1], indices)

        xyz.mul_(40)

        return xyz, feature, indices, lbl
    
    def get_test_item(self, idx):

        pick = idx % self.loop * 5

        idx //= self.loop
        xyz, col, lbl = self.datas[idx]
        
        full_xyz = xyz
        full_lbl = lbl

        indices = grid_subsampling_test(xyz, self.grid_size[0], 2.5 / 14, pick=pick)
        xyz = xyz[indices]
        col = col[indices].float()

        full_nn = KDTree(xyz).knn(full_xyz, 1)[0].squeeze(-1)
    
        col.mul_(1 / 250.)

        xyz -= xyz.min(dim=0)[0]
        feature = torch.cat([col, xyz[:, 2:]], dim=1)

        indices = []
        self.knn(xyz, self.grid_size[::-1], self.k[::-1], indices)    

        xyz.mul_(40)
        
        return xyz, feature, indices, full_nn, full_lbl
    
    def knn(self, xyz: torch.Tensor, grid_size: list, k: list, indices: list, full_xyz: torch.Tensor=None):
        """
        presubsampling and knn search \\
        return indices: knn1, sub1, knn2, sub2, knn3, back_knn1, back_knn2
        """
        first = full_xyz is None
        last = len(k) == 1

        gs = grid_size.pop()
        if first:
            full_xyz = xyz
        else:
            if self.warmup:
                sub_indices = torch.randperm(xyz.shape[0])[:int(xyz.shape[0] / gs)].contiguous()
            else:
                sub_indices = grid_subsampling(xyz, gs)
            xyz = xyz[sub_indices]
            indices.append(sub_indices)

        kdt = KDTree(xyz)
        indices.append(kdt.knn(xyz, k.pop(), False)[0])

        if not last:
            self.knn(xyz, grid_size, k, indices, full_xyz)

        if not first:
            back = kdt.knn(full_xyz, 1, False)[0].squeeze(-1)
            indices.append(back)

        return

def fix_indices(indices: list[torch.Tensor], cnt1: list, cnt2: list):
    """
    fix so ok for indexing as a whole
    """
    first = len(cnt2) == 0
    last = len(cnt1) == 1 or  (len(cnt1) == 2 and len(cnt2) != 0)

    if first:
        c1 = cnt1[-1]
    else:
        indices.pop().add_(cnt1.pop())
        c1 = cnt1[-1]

    knn = indices.pop()
    knn.add_(c1)

    cnt2.append(c1 + knn.shape[0])

    if not last:
        fix_indices(indices, cnt1, cnt2)
    
    if not first:
        indices.pop().add_(c1)


def s3dis_collate_fn(batch):
    """
    [[xcil], [xcil], ...]
    """
    xyz, col, indices, lbl = list(zip(*batch))

    depth = (len(indices[0]) + 2) // 3
    cnt1 = [0] * depth
    pts = []

    for ids in indices:
        pts.extend(x.shape[0] for x in ids[:2*depth:2])
        cnt2 = []
        fix_indices(ids[::-1], cnt1[::-1], cnt2)
        cnt1 = cnt2
    
    xyz = torch.cat(xyz, dim=0)
    col = torch.cat(col, dim=0)
    lbl = torch.cat(lbl, dim=0)
    indices = [torch.cat(ids, dim=0) for ids in zip(*indices)]
    pts = torch.tensor(pts, dtype=torch.int64).view(-1, depth).transpose(0, 1).contiguous()

    return xyz, col, indices, pts, lbl

def s3dis_test_collate_fn(batch):
    return batch[0]