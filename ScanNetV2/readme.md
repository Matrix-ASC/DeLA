### Preparation

Download [ScanNet](http://www.scan-net.org/) v2.

Only files suffixed _vh_clean_2.ply and _vh_clean_2.labels.ply are needed.

Set the dataset path in line 9 of config.py. 

Set processed dataset path in line 11-13 if you want.

Then run prepare_scannetv2.py to process raw data into tensors.

```bash
python prepare_scannetv2.py
```