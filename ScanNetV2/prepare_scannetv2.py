import multiprocessing as mp
import plyfile
import torch
import torch.nn.functional as F
from config import raw_data_path as src, processed_data_path as dest

print(f"Processed data will be saved in:\n{dest}")

dest.mkdir(exist_ok=True)

remapper = torch.zeros(256, dtype=torch.uint8) + 20
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

def calc_normal(xyz, face):
    x0 = xyz[face[:, 0]]
    x1 = xyz[face[:, 1]].sub_(x0)
    x2 = xyz[face[:, 2]].sub_(x0)
    fn = torch.cross(x1, x2, dim=1)
    face = face.view(-1, 1).expand(-1, 3)
    fn = fn.view(-1, 1, 3).repeat(1, 3, 1).view(-1, 3)
    norm = torch.zeros_like(xyz).scatter_add_(dim=0, index=face, src=fn)
    norm = F.normalize(norm, p=2, dim=1, eps=1e-8)
    return norm

def process_scene(entry):
    scene = entry.name
    f0 = dest / f"{scene}.pt"
    if f0.exists():
        print(f"skipping {scene}")
        return
    f1 = entry / f"{scene}_vh_clean_2.ply"
    f2 = entry / f"{scene}_vh_clean_2.labels.ply"
    f1 = plyfile.PlyData().read(f1)
    f2 = plyfile.PlyData().read(f2)
    face = torch.tensor([e[0].tolist() for e in f1.elements[1]])
    f1 = torch.tensor([e.tolist() for e in f1.elements[0]])
    f2 = torch.tensor([e.tolist() for e in f2.elements[0]])
    # check xyz, col in f2 rep label
    assert torch.allclose(f1[:, :3], f2[:, :3])
    xyz = f1[:, :3]
    xyz = xyz - xyz.min(dim=0)[0]
    col = f1[:, 3:6].type(torch.uint8)
    norm = calc_normal(xyz, face)
    label = f2[:, -1].long()
    label = remapper[label]
    torch.save((xyz, col, norm, label), f0)
    print(f"finished {scene}")

entries = [entry for entry in (src / "scans").iterdir() if entry.is_dir()]
p = mp.Pool(processes=8)
p.map(process_scene, entries)
p.close()
p.join()