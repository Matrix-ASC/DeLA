"""
    adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ShapeNetDataLoader.py
"""
import numpy as np
import torch
import random, math
from torch.utils.data import Dataset
import os
import json
from torch.nn import functional as F
from config import data_path, presample_path

class PartTest(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xyz, self.norm, self.shape, self.seg = torch.load(presample_path)
    
    def __len__(self):
        return self.xyz.shape[0]
    
    def __getitem__(self, idx):
        return self.xyz[idx], self.norm[idx], self.shape[idx], self.seg[idx]

class PartNormalDataset(Dataset):
    def __init__(self, npoints=2048, train=True):
        self.train = train
        self.root = data_path
        self.npoints = npoints
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if train:
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            else:
                fns = [fn for fn in fns if fn[0:-4] in test_ids]

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            xyz, norm, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            data = np.loadtxt(fn[1]).astype(np.float32)
            xyz = data[:, 0:3]
            norm = data[:, 3:6]
            seg = data[:, -1]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (xyz, norm, cls, seg)

        if self.train:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            # resample
            xyz = xyz[choice, :]
            norm = norm[choice, :]
            seg = seg[choice]
            
            xyz = torch.from_numpy(xyz).float()
            norm = torch.from_numpy(norm).float()
            seg = torch.from_numpy(seg).long()

            scale = torch.rand((3,)) * 0.4 + 0.8
            xyz *= scale
            if random.random() < 0.2:
                norm.fill_(0.)
            else:
                norm *= scale[[1, 2, 0]] * scale[[2, 0, 1]]
                norm = F.normalize(norm, p=2, dim=-1, eps=1e-8)
            
            jitter = torch.empty_like(xyz).normal_(std=0.001)
            xyz += jitter
        
        xyz = xyz * 40.

        return xyz, norm, cls, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    from pointnet2_ops import pointnet2_utils
    testset = PartNormalDataset(train=False)
    cnt = 0
    xyzs = []
    shapes = []
    segs = []
    normals = []
    for xyz, normal, shape, seg in testset:
        xyz = torch.from_numpy(xyz).cuda().float().unsqueeze(0)
        idx = pointnet2_utils.furthest_point_sample(xyz, 2048).long()
        xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3)).cpu()
        xyzs.append(xyz)
        idx = idx.cpu()
        shapes.append(shape)
        seg = torch.from_numpy(seg).long()[idx.squeeze()]
        segs.append(seg)
        normal = torch.from_numpy(normal).float().unsqueeze(0)
        normal = torch.gather(normal, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        normals.append(normal)
        cnt += 1
        if cnt % 250 == 0:
            print(f"{cnt*100 / len(testset):.1f}% done")
        
    xyzs = torch.cat(xyzs)
    shapes = torch.tensor(shapes, dtype=torch.int64)
    segs = torch.cat(segs).view(len(testset), -1)
    normals = torch.cat(normals)
    print(xyzs.shape, normals.shape, shapes.shape, segs.shape)
    torch.save((xyzs, normals, shapes, segs), presample_path)
