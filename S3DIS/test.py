import torch
from torch import nn
import torch.nn.functional as F
from s3dis import S3DIS, s3dis_test_collate_fn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from delasemseg import DelaSemSeg
from config import s3dis_args, dela_args
from torch.cuda.amp import autocast

torch.set_float32_matmul_precision("high")

loop = 12

testdlr = DataLoader(S3DIS(s3dis_args, partition="5", loop=loop, train=False, test=True), batch_size=1,
                      collate_fn=s3dis_test_collate_fn, pin_memory=True, num_workers=8)

model = DelaSemSeg(dela_args).cuda()

util.load_state("pretrained/best.pt", model=model)

model.eval()

metric = util.Metric(13)
cum = 0
cnt = 0

with torch.no_grad():
    for xyz, feature, indices, nn, y in testdlr:
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            nn = nn.cuda(non_blocking=True).long()
            with autocast():
                p = model(xyz, feature, indices)
            cum = cum + p[nn]
            cnt += 1
            if cnt % loop == 0:
                y = y.cuda(non_blocking=True)
                metric.update(cum, y)
                cnt = cum = 0

metric.print("test: ")
