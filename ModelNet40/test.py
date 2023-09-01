import torch
from torch.cuda.amp import autocast
from ModelNet40 import ModelNet40
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from delacls import DelaCls
from config import dela_args, batch_size

torch.set_float32_matmul_precision("high")

testdlr = DataLoader(ModelNet40(partition="test"), batch_size=batch_size,
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
