import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from shapenetpart import PartTest
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from delapartseg import DelaPartSeg
from config import dela_args, batch_size
import numpy as np
from putil import cls2parts, part_seg_refinement, get_ins_mious

torch.set_float32_matmul_precision("high")

testdlr = DataLoader(PartTest(), batch_size=batch_size,
                      pin_memory=True, num_workers=6)

model = DelaPartSeg(dela_args).cuda()
util.load_state("pretrained/best.pt", model=model)
model.eval()

cls_mious = torch.zeros(16, dtype=torch.float32).cuda(non_blocking=True)
cls_nums = torch.zeros(16, dtype=torch.int32).cuda(non_blocking=True)
ins_miou_list = []

with torch.no_grad():
    for xyz, norm, shape, y in testdlr:
        B, N, _ = xyz.shape
        xyz = xyz.cuda(non_blocking=True)
        shape = shape.cuda(non_blocking=True)
        norm = norm.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        logits = 0
        for i in range(10):
            ixyz = xyz
            inorm = norm
            scale = torch.rand((3,), device=xyz.device) * 0.4 + 0.8
            ixyz = ixyz * scale
            inorm = inorm * (scale[[1, 2, 0]] * scale[[2, 0, 1]])
            inorm = F.normalize(inorm, p=2, dim=-1, eps=1e-8)
            with autocast():
                logits  = logits + model(ixyz, inorm, shape)
        
        p = logits.max(dim=1)[1].view(B, N)
        part_seg_refinement(p, xyz, shape, cls2parts, 10)
        batch_ins_mious = get_ins_mious(p, y, shape, cls2parts)
        ins_miou_list += batch_ins_mious
        for shape_idx in range(B):  # sample_idx
            cur_gt_label = int(shape[shape_idx].cpu().numpy())
            # add the iou belongs to this cat
            cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]
            cls_nums[cur_gt_label] += 1

    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
    for cat_idx in range(16):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)
    with np.printoptions(precision=2, suppress=True):
        print(f'Instance mIoU {ins_miou:.2f}, '
                        f'Class mIoU {cls_miou:.2f}, '
                        f'\n Class mIoUs {cls_mious}')
