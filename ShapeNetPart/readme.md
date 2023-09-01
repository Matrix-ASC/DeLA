### Preparation

Download the dataset. Unzip it. Set data path in line 7 of config.py.

```bash
wget --no-check-certificate https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```

Set presample data path in line 9 of config.py (optional).

Presample test set.

```bash
python shapenetpart.py
```
