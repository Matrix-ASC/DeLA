from types import SimpleNamespace
from torch import nn
import torch
from pathlib import Path

# ModelNet40 dataset path
data_path = Path("xxx/modelnet40_ply_hdf5_2048")

epoch = 600
warmup = 60
batch_size = 32
learning_rate = 2e-3
label_smoothing = 0.2

dela_args = SimpleNamespace()
dela_args.depths = [4, 4, 4]
dela_args.ns = [1024, 256, 64]
dela_args.ks = [20, 20, 20]
dela_args.dims = [96, 192, 384]
dela_args.nbr_dims = [48, 48]  
dela_args.bottleneck = 2048
dela_args.num_classes = 40
drop_path = 0.15
drop_rates = torch.linspace(0., drop_path, sum(dela_args.depths)).split(dela_args.depths)
dela_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
dela_args.bn_momentum = 0.1
dela_args.act = nn.GELU
dela_args.mlp_ratio = 2
dela_args.cor_std = [2.8, 5.3, 10]