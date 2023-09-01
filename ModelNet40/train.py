import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from ModelNet40 import ModelNet40
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
from utils.timm.optim import create_optimizer_v2
import utils.util as util
from delacls import DelaCls
import random, os
from time import time, sleep
from config import dela_args, batch_size, learning_rate as lr, label_smoothing as ls, epoch, warmup

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

traindlr = DataLoader(ModelNet40(), batch_size=batch_size, 
                      shuffle=True, pin_memory=True, 
                      persistent_workers=True, drop_last=True, num_workers=6)
testdlr = DataLoader(ModelNet40(partition="test"), batch_size=batch_size,
                      pin_memory=True, 
                      persistent_workers=True, num_workers=6)

step_per_epoch = len(traindlr)

model = DelaCls(dela_args).cuda()

optimizer = create_optimizer_v2(model, lr=lr, weight_decay=5e-2)
scheduler = CosineLRScheduler(optimizer, t_initial = epoch * step_per_epoch, lr_min = lr/10000,
                                warmup_t=warmup*step_per_epoch, warmup_lr_init = lr/20)
scalar = GradScaler()
# if wish to continue from a checkpoint
resume = False
if resume:
    start_epoch = util.load_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scalar=scalar)["start_epoch"]
else:
    start_epoch = 0

scheduler_step = start_epoch * step_per_epoch

metric = util.Metric(dela_args.num_classes)
ttls = util.AverageMeter() 
best = 0
corls = util.AverageMeter() 

for i in range(start_epoch, epoch):
    model.train()
    ttls.reset()
    corls.reset()
    metric.reset()
    now = time()
    for xyz, y in traindlr:
        lam = scheduler_step/(epoch*step_per_epoch)
        lam = 3e-3 ** lam / 3
        scheduler.step(scheduler_step)
        scheduler_step += 1
        xyz = xyz.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        with autocast():
            p, closs = model(xyz)
            loss = F.cross_entropy(p, y, label_smoothing=ls)
        metric.update(p.detach(), y)
        ttls.update(loss.item())
        corls.update(closs.item())
        optimizer.zero_grad(set_to_none=True)
        loss = loss + closs*lam
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

    print(f"epoch {i}:")
    print(f"loss: {round(ttls.avg, 4)} || cls: {round(corls.avg, 4)}")
    metric.print("train:", iou=False)

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
    print(f"duration: {time() - now}")
    cur = metric.acc
    if best < cur:
        best = cur
        print("new best!")
        util.save_state(f"output/model/{cur_id}/best.pt", model=model)
    
    util.save_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scalar=scalar, start_epoch=i+1)
