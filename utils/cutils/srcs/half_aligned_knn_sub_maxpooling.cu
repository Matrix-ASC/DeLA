#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <torch/extension.h>

constexpr uint64_t group_size = 8;
constexpr uint64_t block = 512;

struct __builtin_align__(group_size*2) halfn
{
    half v[group_size];
};

__global__ void __launch_bounds__(block) half_aligned_knn_edge_maxpooling_forward_kernel(
    half *output,              // BNC
    uint32_t *indices,          // BNC indices for backward
    const half *feature,       // BNC 
    const uint64_t *knn,        // BNk 
    const uint64_t k,
    const uint64_t N, 
    const uint64_t C_,   
    const uint64_t BNC       
){
    // idx = bNC + nC + c
    const uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC) return;
    const uint64_t C = C_ / group_size;
    // bN + n
    const uint64_t BN = idx / C;
    const uint64_t n = BN % N;
    // feature base idx : bNC_ + c*group_size, striding C_
    const uint64_t feature_base = (BN - n) * C_ + (idx % C) * group_size;
    // knn base idx : bNk + nk, striding 1
    uint64_t knn_idx = BN * k;
    const uint64_t knn_end = knn_idx + k;
    uint64_t nbr_idx = knn[knn_idx];
    halfn max_val = *(halfn*)(feature + feature_base + nbr_idx * C_);
    uint32_t max_idx[group_size];
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
    {
        max_idx[f_idx] = nbr_idx;
    }
    for (++knn_idx; knn_idx < knn_end; ++knn_idx)
    {
        nbr_idx = knn[knn_idx];
        const halfn valn = *(halfn*)(feature + feature_base + nbr_idx * C_);
        for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
        {
            const half val = valn.v[f_idx];
            if (__hgt(val, max_val.v[f_idx]))
            {
                max_val.v[f_idx] = val;
                max_idx[f_idx] = nbr_idx;
            }
        }
    }
    const halfn valn = *(halfn*)(feature + feature_base + n * C_);
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
    {
        max_val.v[f_idx] = __hsub(max_val.v[f_idx], valn.v[f_idx]);
        indices[feature_base + n * C_ + f_idx] = max_idx[f_idx];
    }
    *(halfn*)(output + feature_base + n * C_) = max_val;
}


void half_aligned_knn_edge_maxpooling_forward(
    torch::Tensor &output,
    torch::Tensor &indices,
    const torch::Tensor &feature,
    const torch::Tensor &knn
){
    const uint64_t k = knn.size(2);
    const uint64_t N = knn.size(1);
    const uint64_t C = output.size(2);
    const uint64_t BNC = output.size(0) * N * (C / group_size);
    const uint64_t grid = (BNC + block - 1) / block;
    half_aligned_knn_edge_maxpooling_forward_kernel<<<grid, block>>>(
        (half*)output.data_ptr(),
        (uint32_t*)indices.data_ptr(),
        (const half*)feature.data_ptr(),
        (const uint64_t*)knn.data_ptr(),
        k, N, C, BNC
    );
}



__global__ void __launch_bounds__(block) half_aligned_knn_edge_maxpooling_infer_kernel(
    half *output,              // BNC
    const half *feature,       // BNC 
    const uint64_t *knn,        // BNk 
    const uint64_t k,
    const uint64_t N, 
    const uint64_t C_,   
    const uint64_t BNC       
){
    // idx = bNC + nC + c
    const uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC) return;
    const uint64_t C = C_ / group_size;
    // bN + n
    const uint64_t BN = idx / C;
    const uint64_t n = BN % N;
    // feature base idx : bNC_ + c*group_size, striding C_
    const uint64_t feature_base = (BN - n) * C_ + (idx % C) * group_size;
    // knn base idx : bNk + nk, striding 1
    uint64_t knn_idx = BN * k;
    const uint64_t knn_end = knn_idx + k;
    uint64_t nbr_idx = knn[knn_idx];
    halfn max_val = *(halfn*)(feature + feature_base + nbr_idx * C_);
    for (++knn_idx; knn_idx < knn_end; ++knn_idx)
    {
        nbr_idx = knn[knn_idx];
        const halfn valn = *(halfn*)(feature + feature_base + nbr_idx * C_);
        for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
        {
            const half val = valn.v[f_idx];
            if (__hgt(val, max_val.v[f_idx]))
            {
                max_val.v[f_idx] = val;
            }
        }
    }
    const halfn valn = *(halfn*)(feature + feature_base + n * C_);
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
    {
        max_val.v[f_idx] = __hsub(max_val.v[f_idx], valn.v[f_idx]);
    }
    *(halfn*)(output + feature_base + n * C_) = max_val;
}


void half_aligned_knn_edge_maxpooling_infer(
    torch::Tensor &output,
    const torch::Tensor &feature,
    const torch::Tensor &knn
){
    const uint64_t k = knn.size(2);
    const uint64_t N = knn.size(1);
    const uint64_t C = output.size(2);
    const uint64_t BNC = output.size(0) * N * (C / group_size);
    const uint64_t grid = (BNC + block - 1) / block;
    half_aligned_knn_edge_maxpooling_infer_kernel<<<grid, block>>>(
        (half*)output.data_ptr(),
        (const half*)feature.data_ptr(),
        (const uint64_t*)knn.data_ptr(),
        k, N, C, BNC
    );
}




__global__ void half_knn_edge_maxpooling_backward_kernel(
    half *output,              // BNC
    const uint32_t *indices,    // BNC indices for backward
    const half *grad,          // BNC 
    const uint64_t N, 
    const uint64_t C,   
    const uint64_t BNC       
){
    const uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC) return;
    const uint64_t n = idx / C % N;
    const uint64_t backidx = indices[idx];
    const half g = grad[idx];
    const uint64_t high = idx % 2;
    const uint64_t feature_base = idx - n*C + backidx*C - high;
    half2 x;
    x.x = high ? __int2half_rz(0) : g;
    x.y = high ? g : __int2half_rz(0);
    atomicAdd(reinterpret_cast<half2*>(output + feature_base), x);
}

void half_knn_edge_maxpooling_backward(
    torch::Tensor &output,
    const torch::Tensor &indices,
    const torch::Tensor &grad
){
    const uint64_t N = output.size(1);
    const uint64_t C = output.size(2);
    const uint64_t BNC = output.size(0) * N * C;
    const uint64_t grid = (BNC + block - 1) / block;
    half_knn_edge_maxpooling_backward_kernel<<<grid, block>>>(
        (half*)output.data_ptr(),
        (const uint32_t*)indices.data_ptr(),
        (const half*)grad.data_ptr(),
        N, C, BNC
    );
}
