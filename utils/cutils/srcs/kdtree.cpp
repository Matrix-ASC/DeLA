#include <torch/extension.h>
#include "nanoflann.hpp"

struct TensorPointCloudAdaptor{
    const float *pc;
    const size_t pc_size;

    TensorPointCloudAdaptor(const torch::Tensor &pc_)
        : pc((const float*)pc_.data_ptr()), pc_size(pc_.size(0))
    {
    }

    inline size_t kdtree_get_point_count() const
    {
        return pc_size;
    }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return pc[(idx*3 + dim)];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, TensorPointCloudAdaptor>, TensorPointCloudAdaptor, 3>;

std::vector<size_t> kdtree_build(const torch::Tensor &pc, const size_t max_leaf_size)
{
    const TensorPointCloudAdaptor *pca = new TensorPointCloudAdaptor(pc);
    auto kdtree = new KDTree(3, *pca, {max_leaf_size});
    return {(size_t)kdtree, (size_t)pca};
}

void kdtree_free(size_t kdtree, size_t pca)
{
    delete (KDTree*)kdtree;
    delete (TensorPointCloudAdaptor*)pca;
}

void kdtree_knn(size_t kdtree, const torch::Tensor &qpc, torch::Tensor &indices, torch::Tensor &dists, const bool sorted)
{
    const size_t k = indices.size(1);
    const size_t queries = qpc.size(0);
    uint32_t *pindices = (uint32_t*)indices.data_ptr();
    float *pdists = (float*)dists.data_ptr();
    const float *pqpc = (const float*)qpc.data_ptr();
    for (size_t i = 0; i < queries; ++i)
    {
        nanoflann::KNNResultSetHeap<float, uint32_t> result(k, pindices + i*k, pdists + i*k);
        ((KDTree*)kdtree)->findNeighbors(result, pqpc + i*3);
    }
    if (!sorted)
        return;
    for (size_t i = 0; i < queries; ++i)
    {
        nanoflann::KNNResultSetHeap<float, uint32_t> result(k, pindices + i*k, pdists + i*k);
        result.sort();
    }
}
