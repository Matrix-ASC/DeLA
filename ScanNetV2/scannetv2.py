import torch, numpy as np
from torch.nn import functional as F
import random, scipy.interpolate, scipy.ndimage
import math
from torch.utils.data import Dataset
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.cutils import grid_subsampling, KDTree, grid_subsampling_test
from config import processed_data_path, scan_train, scan_val

# adapted from https://github.com/Gofinge/PointTransformerV2/blob/main/pcr/datasets/transform.py
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=False, fill_value=0)
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, coord):
        coord = coord.numpy()
        if random.random() < 0.95:
            for granularity, magnitude in self.distortion_params:
                coord = self.elastic_distortion(coord, granularity, magnitude)
        return torch.from_numpy(coord)

class ScanNetV2(Dataset):
    r"""
    partition   =>   train / val
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
    def __init__(self, args, partition="train", loop=6, train=True, test=False, warmup=False):

        self.paths = scan_train if partition == "train" else scan_val
        self.paths = [processed_data_path / f"{p}.pt" for p in self.paths]

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
        self.els = ElasticDistortion()


    def __len__(self):
        return len(self.paths) * self.loop
    
    def __getitem__(self, idx):
        if self.test:
            return self.get_test_item(idx)

        idx //= self.loop
        xyz, col, norm, lbl = self.datas[idx]

        if self.train:
            angle = random.random() * 2 * math.pi
            cos, sin = math.cos(angle), math.sin(angle)
            rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
            norm = norm @ rotmat
            rotmat *= random.uniform(0.8, 1.2)
            xyz = xyz @ rotmat
            xyz = self.els(xyz)
            xyz -= xyz.min(dim=0)[0]

        # here grid size is assumed 0.02, so estimated downsampling ratio is ~1.5
        if self.train:
            indices = grid_subsampling(xyz, self.grid_size[0], 2.5 / 1.5)
        else:
            indices = grid_subsampling_test(xyz, self.grid_size[0], 2.5 / 1.5, pick=0)

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
        norm = norm[indices]
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
        if self.train and random.random() < 0.2:
            norm.fill_(0.)
        
        height = xyz[:, 2:]
        feature = torch.cat([col, height, norm], dim=1)

        indices = []
        self.knn(xyz, self.grid_size[::-1], self.k[::-1], indices)

        xyz.mul_(60)

        return xyz, feature, indices, lbl
    
    def get_test_item(self, idx):
        rotations = [0, 0.5, 1, 1.5]
        scales = [0.95, 1, 1.05]
        augs = len(rotations) * len(scales)
        aug = idx % self.loop
        pick = aug // augs
        aug %= augs

        idx //= self.loop
        xyz, col, norm, lbl = self.datas[idx]

        angle = math.pi * rotations[aug // len(scales)]
        cos, sin = math.cos(angle), math.sin(angle)
        rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        norm = norm @ rotmat
        rotmat *= scales[aug % len(scales)]
        xyz = xyz @ rotmat
        xyz -= xyz.min(dim=0)[0]
        
        full_xyz = xyz
        full_lbl = lbl

        indices = grid_subsampling_test(xyz, self.grid_size[0], 2.5 / 1.5, pick=pick)
        xyz = xyz[indices]
        col = col[indices].float()
        norm = norm[indices]

        full_nn = KDTree(xyz).knn(full_xyz, 1)[0].squeeze(-1)
    
        col.mul_(1 / 250.)

        xyz -= xyz.min(dim=0)[0]
        feature = torch.cat([col, xyz[:, 2:], norm], dim=1)

        indices = []
        self.knn(xyz, self.grid_size[::-1], self.k[::-1], indices)    

        xyz.mul_(60)
        
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
            indices.append(kdt.knn(full_xyz, 1, False)[0].squeeze(-1))

        return

def fix_indices(indices, cnt1: list, cnt2: list):
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


def scan_collate_fn(batch):
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

def scan_test_collate_fn(batch):
    return batch[0]

