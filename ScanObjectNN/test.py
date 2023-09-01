import torch
import torch.nn as nn
from ScanObjectNN import ScanObjectNN
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from delacls import DelaCls
from config import dela_args, batch_size
from torch.cuda.amp import autocast

torch.set_float32_matmul_precision("high")

testdlr = DataLoader(ScanObjectNN(partition="test"), batch_size=batch_size,
                      pin_memory=True, num_workers=6)

model = DelaCls(dela_args).cuda()
util.load_state("pretrained/best.pt", model=model)

metric = util.Metric(dela_args.num_classes)
model.eval()
metric.reset()
with torch.no_grad():
    for xyz, y in testdlr:
        xyz = xyz.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        with autocast():
            p = model(xyz)
        metric.update(p, y)

metric.print("val:  ", iou=False)
