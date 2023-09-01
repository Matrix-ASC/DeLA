from types import SimpleNamespace
from pathlib import Path
from torch import nn
import torch

# ShapeNetPart dataset path
data_path = Path("xxx/shapenetcore_partanno_segmentation_benchmark_v0_normal")

presample_path = data_path.parent / "shapenet_part_presample.pt"

epoch = 250
warmup = 20
batch_size = 32
learning_rate = 2e-3
label_smoothing = 0.2

dela_args = SimpleNamespace()
dela_args.depths = [4, 4, 4, 4]
dela_args.ns = [2048, 512, 192, 64]
dela_args.ks = [20, 20, 20, 20]
dela_args.dims = [96, 192, 320, 512]
dela_args.nbr_dims = [48,48]  
dela_args.head_dim = 320
dela_args.num_classes = 50
dela_args.shape_classes = 16
drop_path = 0.15
drop_rates = torch.linspace(0., drop_path, sum(dela_args.depths)).split(dela_args.depths)
dela_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
dela_args.head_drops = torch.linspace(0., 0.15, len(dela_args.depths)).tolist()
dela_args.bn_momentum = 0.1
dela_args.act = nn.GELU
dela_args.mlp_ratio = 2
dela_args.cor_std = [0.75, 1.5, 2.5, 4.7]