### Preparation

Download [S3DIS](http://buildingparser.stanford.edu/dataset.html) and unzip it. We use Stanford3dDataset_v1.2_Aligned_Version.

Set the dataset path according to prepare_s3dis.py (don't run it, read the top lines).

Then run prepare_s3dis.py to process raw data into tensors. Fix illegal characters and rerun if necessary.

```bash
python prepare_s3dis.py
```