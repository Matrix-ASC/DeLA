#include <torch/extension.h>
#include <random>

struct Cell
{
    uint64_t key;
    uint32_t idx;
    uint32_t weight;
    uint32_t running_weight;
    uint32_t count;
    Cell *next;
};

struct SlowHash
{
    Cell **table;
    Cell *storage;
    uint64_t size;
    uint64_t multiplier;
    uint64_t count;
    uint32_t pick;
    std::mt19937 rand;

    SlowHash(Cell **table, Cell *storage, const uint64_t size, const uint32_t pick)
        : table(table), storage(storage), size(size), multiplier(~0ull / size + 1), count(0), pick(pick)
    {
        rand.seed(*(uint32_t*)torch::randint(0, std::numeric_limits<uint32_t>::max(), 1, torch::kInt64).data_ptr());
    }

    void insert(const uint64_t key, const uint32_t idx)
    {
        const uint64_t slow_idx = __uint128_t(key * multiplier) * size >> 64;
        Cell **cell = &table[slow_idx];
        for (; *cell && (*cell)->key != key; cell = &(*cell)->next);
        if (*cell == nullptr)
        {
            *cell = storage + count;
            ++count;
            (*cell)->next = nullptr;
            (*cell)->key = key;
            (*cell)->weight = (*cell)->running_weight = rand();
            (*cell)->count = 0;
            (*cell)->idx = idx;
            if (pick == 0)  (*cell)->weight = std::numeric_limits<uint32_t>::max();
        } else {
            if (++(*cell)->count == pick)
            {
                (*cell)->weight = std::numeric_limits<uint32_t>::max();
                (*cell)->idx = idx;
                return;
            }
            uint32_t new_weight = (*cell)->running_weight * (uint32_t)31 + (uint32_t)1857864947;
            if (new_weight > (*cell)->weight)
            {
                (*cell)->weight = new_weight;
                (*cell)->idx = idx;
            }
            (*cell)->running_weight = new_weight;
        }
    }

    torch::Tensor finalize()
    {
        auto indices = torch::empty(count, torch::kInt64);
        uint64_t *pindices = (uint64_t*)indices.data_ptr();
        for (uint64_t i = 0; i < count; ++i)
            pindices[i] = storage[i].idx;
        return indices;
    }
};

torch::Tensor grid_subsampling_test(const torch::Tensor &pc_, const float grid_size_, torch::Tensor &hash_table, torch::Tensor &hash_storage, uint32_t pick)
{
    const double grid_size = 1 / (double)grid_size_;
    const uint64_t slow_hash_size = hash_table.size(0);
    SlowHash slow_hash((Cell**)hash_table.data_ptr(), (Cell*)hash_storage.data_ptr(), slow_hash_size, pick);
    const uint64_t pc_size = pc_.size(0);
    const float *pc = (const float*)pc_.data_ptr();
    const uint64_t x_step = ((1ull << 42) / slow_hash_size * slow_hash_size + uint64_t(slow_hash_size * 0.292135));
    const uint64_t z_step = ((1ull << 21) / slow_hash_size * slow_hash_size + uint64_t(slow_hash_size * 0.559518));
    const auto d2u = [](double x)
    {
        int64_t tmp = x;
        return *(uint64_t*)&tmp;
    };
    for (uint64_t i = 0; i < pc_size; ++i)
    {
        const uint64_t key = (d2u(pc[i*3]*grid_size) * x_step) + (d2u(pc[i*3 + 1]*grid_size)) + (d2u(pc[i*3 + 2]*grid_size) * z_step);
        slow_hash.insert(key, i);
    }
    return slow_hash.finalize();
}
