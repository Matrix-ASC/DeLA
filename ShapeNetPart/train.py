import torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from shapenetpart import PartNormalDataset, PartTest
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
from utils.timm.optim import create_optimizer_v2
import utils.util as util
from delapartseg import DelaPartSeg
import random, os
from time import time, sleep
from config import dela_args, batch_size, learning_rate as lr, label_smoothing as ls, epoch, warmup
from putil import cls2parts, get_ins_mious

torch.set_float32_matmul_precision("high")

cur_id = "01"
os.makedirs(f"output/log/{cur_id}", exist_ok=True)
os.makedirs(f"output/model/{cur_id}", exist_ok=True)
logfile = f"output/log/{cur_id}/out.log"
errfile = f"output/log/{cur_id}/err.log"
logfile = open(logfile, "a", 1)
errfile = open(errfile, "a", 1)
sys.stdout = logfile
sys.stderr = errfile

print(r"base")

traindlr = DataLoader(PartNormalDataset(), batch_size=batch_size, 
                      shuffle=True, pin_memory=True, 
                      persistent_workers=True, drop_last=True, num_workers=8)
testdlr = DataLoader(PartTest(), batch_size=batch_size,
                      pin_memory=True, 
                      persistent_workers=True, num_workers=8)

step_per_epoch = len(traindlr)

model = DelaPartSeg(dela_args).cuda()

optimizer = create_optimizer_v2(model, lr=lr, weight_decay=5e-2)
scheduler = CosineLRScheduler(optimizer, t_initial = epoch * step_per_epoch, lr_min = lr/10000,
                                warmup_t=warmup*step_per_epoch, warmup_lr_init = lr/20)
scaler = GradScaler()
# if wish to continue from a checkpoint
resume = False
if resume:
    start_epoch = util.load_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler)["start_epoch"]
else:
    start_epoch = 0

scheduler_step = start_epoch * step_per_epoch

ttls = util.AverageMeter() 
corls = util.AverageMeter() 
best = 0

for i in range(start_epoch, epoch):
    model.train()
    ttls.reset()
    corls.reset()
    now = time()
    for xyz, norm, shape, y in traindlr:
        lam = scheduler_step/(epoch*step_per_epoch)
        lam = 3e-3 ** lam * .25
        scheduler.step(scheduler_step)
        scheduler_step += 1
        xyz = xyz.cuda(non_blocking=True)
        shape = shape.cuda(non_blocking=True)
        norm = norm.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        with autocast():
            p, idx, closs = model(xyz, norm, shape)
            y = torch.gather(y, 1, idx[0])
            y[idx[1]] = 255
            y = y.flatten()
            loss = F.cross_entropy(p, y, label_smoothing=ls, ignore_index=255)
        ttls.update(loss.item())
        corls.update(closs.item())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss + closs*lam).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"epoch {i}:")
    print(f"loss: {round(ttls.avg, 4)} || cls: {round(corls.avg, 4)}")

    model.eval()
    cls_mious = torch.zeros(16, dtype=torch.float32, device="cuda")
    cls_nums = torch.zeros(16, dtype=torch.int32, device="cuda")
    ins_miou_list = []
    with torch.no_grad():
        for xyz, norm, shape, y in testdlr:
            B, N, _ = xyz.shape
            xyz = xyz.cuda(non_blocking=True)
            shape = shape.cuda(non_blocking=True)
            norm = norm.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with autocast():
                p = model(xyz, norm, shape).max(dim=1)[1].view(B, N)
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
    cls_mious = [round(cm, 2) for cm in cls_mious.tolist()]
    
    with np.printoptions(precision=2, suppress=True):
        print(f'Instance mIoU {ins_miou:.2f}, '
                        f'Class mIoU {cls_miou:.2f}, '
                        f'\n Class mIoUs {cls_mious}')
    
    print(f"duration: {time() - now}")
    cur = ins_miou
    # ins miou can be premature, and we check the last 50 epochs
    if best <= cur and i >= 200:
        best = cur
        print("new best!")
        util.save_state(f"output/model/{cur_id}/best.pt", model=model)
    
    util.save_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler, start_epoch=i+1)
