from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from torch import nn
import torch

# ScanNetV2 dataset path
# should contain scans/
raw_data_path = Path("/xxx/")

processed_data_path = raw_data_path.parent / "scannetv2"
# if you want to set the processed dataset path, uncomment here
#processed_data_path = Path("")

scan_train = Path(__file__).parent / "scannetv2_train.txt"
scan_val = Path(__file__).parent / "scannetv2_val.txt"
with open(scan_train, 'r') as file:
    scan_train = [line.strip() for line in file.readlines()]
with open(scan_val, 'r') as file:
    scan_val = [line.strip() for line in file.readlines()]

epoch = 100
warmup = 10
batch_size = 8
learning_rate = 6e-3
label_smoothing = 0.2

scan_args = SimpleNamespace()
scan_args.k = [24, 24, 24, 24, 24]
scan_args.grid_size = [0.02, 0.04, 0.08, 0.16, 0.32]

scan_args.max_pts = 80000

scan_warmup_args = deepcopy(scan_args)
scan_warmup_args.grid_size = [0.02, 2, 3.5, 3.5, 4]

dela_args = SimpleNamespace()
dela_args.ks = scan_args.k
dela_args.depths = [4, 4, 4, 8, 4]
dela_args.dims = [64, 96, 160, 288, 512]
dela_args.nbr_dims = [32, 32]
dela_args.head_dim = 288
dela_args.num_classes = 20
drop_path = 0.1
drop_rates = torch.linspace(0., drop_path, sum(dela_args.depths)).split(dela_args.depths)
dela_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
dela_args.head_drops = torch.linspace(0., 0.2, len(dela_args.depths)).tolist()
dela_args.bn_momentum = 0.02
dela_args.act = nn.GELU
dela_args.mlp_ratio = 2
# gradient checkpoint
dela_args.use_cp = False

dela_args.cor_std = [1.6, 2.5, 5, 10, 20]