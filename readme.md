# Decoupled Local Aggregation for Point Cloud Learning



Here is the PyTorch implementation of our [paper](https://arxiv.org/abs/2308.16532).

Please feel free to open an issue if you have any questions or suggestions.

[DeLA v2](https://github.com/Matrix-ASC/DeLA_v2) is now available with efficiency and expressiveness improvements.


## Contents

### grid_subsampling.md 

Details the sampling algorithm we use.

### utils 

**cutils** contains cpp/cuda functions (neighbor edge-max-pooling, grid subsampling, knn search).

**util.py** has some python functions (state saving/loading, metric calculation).

**timm** is a tiny [timm](https://github.com/rwightman/pytorch-image-models). Only functions used are kept to reduce dependencies.

**pointnet2_ops_lib** is from [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).

### other directories

Dataset specific implementation.

Basically all contain: train, test, config, model, dataset, readme.md and pretrained (folder containing training log and trained model).

Please be aware that the code used to train our pretrained models has been (and likely will be) slightly amended, so you may notice discrepancies in the training log. (currently only fixed reset of corls)



## Dependency

We list libraries (tested version) and the way we install it for reference.

We believe newer versions are generally fine and there's no need to follow this guide exactly.

### General dependency

Create and activate environment using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

Add [conda-forge](https://conda-forge.org/index.html) channel to the environment first and comment the original channels.

Switch mirrors (channel sources) if necessary. 

```bash
conda create -n dela python=3.10

conda activate dela
```

Install pytorch.

Specify a version with cuda.

```bash
conda install pytorch=1.13.1=cuda112py310he33e0d6_200
```

Other tools should be automatically installed in this step, e.g., numpy (1.24.3), cudatoolkit (11.8.0), ...

An important thing to note is, as of 2023.5, under this setting, llvm-openmp >= 16.0.0 can cause a problem with pytorch's multi threaded dataloading.

We currently do not know the exact cause, and simply use version 15.

```bash
conda install llvm-openmp=15
```

Then check if cuda kernels can be compiled smoothly.

Remember to set CUDA_HOME and TORCH_CUDA_ARCH_LIST properly.

```bash
python
Python 3.10.11 | packaged by conda-forge | (main, May 10 2023, 18:58:44) [GCC 11.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from utils.cutils import knn_edge_maxpooling
```

Compilation verbose is set to True and can be disabled in line 15 of cutils' init file if this step goes fine.

### Classification dependency

Install h5py.

```bash
conda install h5py=3.8.0
```

Install pointnet2_ops.

```bash
cd utils/pointnet2_ops_lib/
pip install .
```

### Part segmentation dependency

Install pointnet2_ops.

```bash
cd utils/pointnet2_ops_lib/
pip install .
```

### ScanNet v2 dependency

Install plyfile and scipy.

```bash
conda install plyfile=0.8.1 scipy=1.10.1
```



## Train & test

To train & test on a specific dataset, first enter corresponding directory and **set up the dataset following readme.md**.

### Train

```bash
python train.py& 
disown
```

Saved logs and models are under output/.

Check train.py and other scripts for more details. 

To resume from a checkpoint, follow train.py (search for "resume").

### Test

```bash
python test.py
```

As random seeds are not fixed, output for segmentation is slightly unstable.

The results of pretrained models should be near reported ones with high probability.



## Performance & Training time and memory requirement

Here we list mean $\pm$ std of 3 random runs.

Training time is measured on Ubuntu 22.04 with an RTX 4090 GPU and a 13600k CPU.

Memory consumption is checked from nvidia-smi with pytorch 1.13.1. Newer pytorch versions may slightly save memory.

### S3DIS

mIoU:       73.48 $\pm$ 0.46

Time:       around 110 min, 90 min (parallelize 2 jobs), 140 min (with gradient checkpoint), 115 min (parallel + checkpoint)

Memory:     9496 MB, 6452 MB (with gradient checkpoint)

### ScanNet v2

mIoU:       75.88 $\pm$ 0.07

Time:       around 320 min, 410 min (with gradient checkpoint)

Memory:     21430 MB, 13152 MB (with gradient checkpoint)

### ShapeNetPart

Results without voting:

ins mIoU:   86.94 $\pm$ 0.08

cat mIoU:   85.41 $\pm$ 0.39

Time:       around 80 min

Memory:     5004 MB

### ScanObjectNN

OA:         90.15 $\pm$ 0.25

mAcc:       89.01 $\pm$ 0.39

Time:       around 30 min

Memory:     2552 MB

### ModelNet40

OA:         93.75 $\pm$ 0.18

mAcc:       91.15 $\pm$ 0.61

Time:       around 50 min

Memory:     2594 MB
