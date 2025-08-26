from pathlib import Path
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
from torch.nn import functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import os

path = Path(__file__).parent
build_dir = path / "build"
build_dir.mkdir(exist_ok=True)
sources = [str(p) for p in path.glob("srcs/*.*") if p.suffix in [".cpp", ".cu"]]

cutils = load("cutils_", sources=sources, extra_cflags=["-O3", "-mavx2", "-funroll-loops"], extra_cuda_cflags=["-Xptxas","-v"],
              verbose=True, build_directory=build_dir)

def next_prime(x) -> int:
    r"""
    Finds the next prime, x included.           
    x should be >= 3 for a correct result.
    """
    x = int(x) | 1
    for i in range(x, 2*x, 2):
        prime = True
        for j in range(3, int(i**0.5) + 1, 2):
            if i % j == 0:
                prime = False
                break
        if prime:
            return i

def grid_subsampling(xyz: torch.Tensor, grid_size: float, hash_size: float=1.) -> torch.Tensor:
    r"""
    xyz: N x 3, float, non-negative coordinates
    grid_size: float, positive
    hash_size: How large the hash table should be relative to the original point cloud size.
                If estimated downsampling ratio is k, i.e., ori_size = k * subsampled_size,
                then recommended value is 2~3 / k.
                Must be greater than 1 / real_k
    return value: M, int64, selected indices
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
    if xyz.stride(0) != 3:
        xyz = xyz.contiguous()
    size = xyz.shape[0] * hash_size
    size = next_prime(size + 1)
    table = torch.zeros((size,), dtype=torch.int64)
    storage = torch.empty((size * 3,), dtype=torch.int64)
    indices = cutils.grid_subsampling(xyz, grid_size, table, storage)
    return indices

def grid_subsampling_test(xyz: torch.Tensor, grid_size: float, hash_size: float=1., pick=0) -> torch.Tensor:
    r"""
    xyz: N x 3, float, non-negative coordinates
    grid_size: float, positive
    hash_size: How large the hash table should be relative to the original point cloud size.
                If estimated downsampling ratio is k, i.e., ori_size = k * subsampled_size,
                then recommended value is 2~3 / k.
                Must be greater than 1 / real_k
    pick:  the nth point in the same grid to pick, random picked if actual resident points < pick
    return value: M, int64, selected indices
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
    if xyz.stride(0) != 3:
        xyz = xyz.contiguous()
    size = xyz.shape[0] * hash_size
    size = next_prime(size + 1)
    table = torch.zeros((size,), dtype=torch.int64)
    storage = torch.empty((size * 4,), dtype=torch.int64)
    indices = cutils.grid_subsampling_test(xyz, grid_size, table, storage, pick)
    return indices

class KDTree():
    r"""
    kdt = KDTree(xyz) 
    indices, squared_dists = kdt.knn(query_xyz, k=16, ordered=True)
    indices: int32
    dists: float

    Setting ordered = False (default) can be 1.1-1.2x faster. 
    If there are not enough neighbors, the nearest point is used for padding. 
    Resources (reference to xyz, built tree) are freed when kdt goes out of life scope.
    """
    def __init__(self, xyz: torch.Tensor, max_leaf_size=20):
        assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
        if xyz.stride(0) != 3:
            xyz = xyz.contiguous()
        # reserve xyz for knn search
        self.xyz = xyz
        self.n = xyz.shape[0]
        self.tree, self.pca = cutils.kdtree_build(xyz, max_leaf_size)
    
    def __del__(self):
        cutils.kdtree_free(self.tree, self.pca)
    
    def knn(self, query: torch.Tensor, k=1, ordered=False):
        assert query.ndim == 2 and query.shape[1] == 3 and query.dtype == torch.float
        if query.stride(0) != 3:
            query = query.contiguous()
        queries = query.shape[0]
        nbrs = min(self.n, k)
        if self.n < k : ordered = True
        indices = torch.empty((queries, nbrs), dtype=torch.int32)
        dists = torch.empty((queries, nbrs), dtype=torch.float)
        cutils.kdtree_knn(self.tree, query, indices, dists, ordered)
        if self.n < k:
            indices = torch.cat([indices, indices[:, :1].expand(-1, k - self.n)], dim=1)
            dists = torch.cat([dists, dists[:, :1].expand(-1, k - self.n)], dim=1)
        return indices, dists

class KEMP(Function):
    r"""
    f_i = max{f_j | j in knn_i} - f_i
    output = knn_edge_maxpooling(feature, knn, training=True)  

    Only cuda version supported.

    feature: BNC, float / half
    knn:     BNk, int64
    output:  BNC, float / half

    While not training and gradient is not required, 
    backward indices are not saved. Consumed time and space reduced slightly.
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, feature: torch.Tensor, knn: torch.Tensor, training: bool=True) -> torch.Tensor:
        assert feature.is_cuda and knn.is_cuda
        assert feature.is_contiguous() and knn.is_contiguous() and feature.shape[:2] == knn.shape[:2]
        assert knn.dtype == torch.int64
        if feature.dtype == torch.half:
            assert feature.shape[-1] % 8 == 0, "KEMP half precision impl only supports multiples of 8 as feature dim"
        elif feature.dtype == torch.float32:
            assert feature.shape[-1] % 4 == 0, "KEMP single precision impl only supports multiples of 4 as feature dim"
        else:
            raise NotImplementedError

        output = torch.empty_like(feature)
        if training or feature.requires_grad:
            indices = torch.empty_like(feature, dtype=torch.int32)
            if feature.dtype == torch.half:
                cutils.half_aligned_knn_edge_maxpooling_forward(output, indices, feature, knn)  
            else:
                cutils.aligned_knn_edge_maxpooling_forward(output, indices, feature, knn)
            ctx.save_for_backward(indices)
        else:
            if feature.dtype == torch.half:
                cutils.half_aligned_knn_edge_maxpooling_infer(output, feature, knn)
            else:
                cutils.aligned_knn_edge_maxpooling_infer(output, feature, knn)
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        output = -grad
        indices, = ctx.saved_tensors
        if grad.dtype == torch.half:
            cutils.half_knn_edge_maxpooling_backward(output, indices, grad)  
        else: 
            cutils.knn_edge_maxpooling_backward(output, indices, grad)
        return output, None, None

knn_edge_maxpooling = KEMP.apply
