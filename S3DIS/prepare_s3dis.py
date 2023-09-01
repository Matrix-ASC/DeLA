"""
    code adapted from https://github.com/aRI0U/RandLA-Net-pytorch/blob/master/utils/prepare_s3dis.py

    1. set raw dataset path  xxx/Stanford3dDataset_v1.2_Aligned_Version   in line 8 of config.py
    2. set processed dataset path (default: xxx/s3dis)  in line 10-12 of config.py
    3. run the code, if sth goes wrong, follow the error message, 
       locate and replace the illegal character with a space " ",
       re-run
       
       sample err msg:
       ValueError: the number of columns changed from 6 to 5 at row 180389; use `usecols` to select a subset and avoid this error
       use this and preceding printed filename to locate room, object, row
"""

import torch
import numpy as np
from pathlib import Path
import warnings
from config import raw_data_path, processed_data_path
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

print(f"Processed data will be saved in:\n{processed_data_path}")

processed_data_path.mkdir(exist_ok=True)

labels_dict = {
  "ceiling": 0,
  "floor": 1,
  "wall": 2,
  "beam": 3,
  "column": 4,
  "window": 5,
  "door": 6,
  "table": 7,
  "chair": 8,
  "sofa": 9,
  "bookcase": 10,
  "board": 11,
  "clutter": 12,
  "stairs": 12
}

for area_number in range(1,7):
    print(f'Reencoding point clouds of area {area_number:d}')
    dir = raw_data_path / f'Area_{area_number:d}'
    if not dir.exists():
        warnings.warn(f'Area {area_number:d} not found')
        continue
    for pc_path in sorted(list(dir.iterdir())):
        if not pc_path.is_dir:
            continue
        pc_name = f'{area_number:d}_' + pc_path.stem + '.pt'
        pc_file = processed_data_path / pc_name

        if pc_file.exists():
            continue

        points_xyz = []
        points_col = []
        points_lbl = []
        for elem in sorted(list(pc_path.glob('Annotations/*.txt'))):
            label = elem.stem.split('_')[0]
            try:
                points = torch.from_numpy(np.loadtxt(elem, dtype=np.float32))
            except Exception as e:
                print(elem)
                raise e
            label_id = labels_dict[label]
            points_xyz.append(points[:, :3])
            points_col.append(points[:, 3:])
            points_lbl.append(torch.full((points.shape[0],), label_id, dtype=torch.uint8))

        if points_xyz == []:
            continue

        points_xyz = torch.cat(points_xyz, dim=0)
        points_xyz = points_xyz - points_xyz.min(dim=0)[0]
        points_col = torch.cat(points_col, dim=0).type(torch.uint8)
        points_lbl = torch.cat(points_lbl, dim=0)

        torch.save((points_xyz, points_col, points_lbl), pc_file)

print('Done.')
