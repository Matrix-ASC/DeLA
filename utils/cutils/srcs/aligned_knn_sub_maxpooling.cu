#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <torch/extension.h>

constexpr uint64_t group_size = 4;
constexpr uint64_t block = 512;

struct __builtin_align__(group_size*4) floatn
{
    float v[group_size];
};

__global__ void __launch_bounds__(block) aligned_knn_edge_maxpooling_forward_kernel(
    float *output,              // BNC
    uint32_t *indices,          // BNC indices for backward
    const float *feature,       // BNC 
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
    floatn max_val = *(floatn*)(feature + feature_base + nbr_idx * C_);
    uint32_t max_idx[group_size];
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
    {
        max_idx[f_idx] = nbr_idx;
    }
    for (++knn_idx; knn_idx < knn_end; ++knn_idx)
    {
        nbr_idx = knn[knn_idx];
        const floatn valn = *(floatn*)(feature + feature_base + nbr_idx * C_);
        for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
        {
            const float val = valn.v[f_idx];
            if (val > max_val.v[f_idx])
            {
                max_val.v[f_idx] = val;
                max_idx[f_idx] = nbr_idx;
            }
        }
    }
    const floatn valn = *(floatn*)(feature + feature_base + n * C_);
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
    {
        max_val.v[f_idx] -= valn.v[f_idx];
        indices[feature_base + n * C_ + f_idx] = max_idx[f_idx];
    }
    *(floatn*)(output + feature_base + n * C_) = max_val;
}


void aligned_knn_edge_maxpooling_forward(
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
    aligned_knn_edge_maxpooling_forward_kernel<<<grid, block>>>(
        (float*)output.data_ptr(),
        (uint32_t*)indices.data_ptr(),
        (const float*)feature.data_ptr(),
        (const uint64_t*)knn.data_ptr(),
        k, N, C, BNC
    );
}



__global__ void __launch_bounds__(block) aligned_knn_edge_maxpooling_infer_kernel(
    float *output,              // BNC
    const float *feature,       // BNC 
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
    floatn max_val = *(floatn*)(feature + feature_base + nbr_idx * C_);
    for (++knn_idx; knn_idx < knn_end; ++knn_idx)
    {
        nbr_idx = knn[knn_idx];
        const floatn valn = *(floatn*)(feature + feature_base + nbr_idx * C_);
        for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
        {
            const float val = valn.v[f_idx];
            if (val > max_val.v[f_idx])
            {
                max_val.v[f_idx] = val;
            }
        }
    }
    const floatn valn = *(floatn*)(feature + feature_base + n * C_);
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
    {
        max_val.v[f_idx] -= valn.v[f_idx];
    }
    *(floatn*)(output + feature_base + n * C_) = max_val;
}


void aligned_knn_edge_maxpooling_infer(
    torch::Tensor &output,
    const torch::Tensor &feature,
    const torch::Tensor &knn
){
    const uint64_t k = knn.size(2);
    const uint64_t N = knn.size(1);
    const uint64_t C = output.size(2);
    const uint64_t BNC = output.size(0) * N * (C / group_size);
    const uint64_t grid = (BNC + block - 1) / block;
    aligned_knn_edge_maxpooling_infer_kernel<<<grid, block>>>(
        (float*)output.data_ptr(),
        (const float*)feature.data_ptr(),
        (const uint64_t*)knn.data_ptr(),
        k, N, C, BNC
    );
}

__global__ void knn_edge_maxpooling_backward_kernel(
    float *output,              // BNC
    const uint32_t *indices,    // BNC indices for backward
    const float *grad,          // BNC 
    const uint64_t N, 
    const uint64_t C,   
    const uint64_t BNC       
){
    const uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC) return;
    const uint64_t n = idx / C % N;
    const uint64_t feature_base = idx - n*C;
    const uint64_t backidx = indices[idx];
    const float g = grad[idx];
    atomicAdd(output + feature_base + backidx*C, g);
}

void knn_edge_maxpooling_backward(
    torch::Tensor &output,
    const torch::Tensor &indices,
    const torch::Tensor &grad
){
    const uint64_t N = output.size(1);
    const uint64_t C = output.size(2);
    const uint64_t BNC = output.size(0) * N * C;
    const uint64_t grid = (BNC + block - 1) / block;
    knn_edge_maxpooling_backward_kernel<<<grid, block>>>(
        (float*)output.data_ptr(),
        (const uint32_t*)indices.data_ptr(),
        (const float*)grad.data_ptr(),
        N, C, BNC
    );
}