#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T>
__device__ T reduce_sum(cg::thread_block_tile<32>& group, T val) {
    for (int offset = group.size()/2; offset > 0; offset /= 2) {
        T temp = group.shfl_down(val, offset);
        val += temp;
    }
    return val;
}

template <typename T>
__device__ T reduce_max(cg::thread_block_tile<32>& group, T val) {
    for (int offset = group.size()/2; offset > 0; offset /= 2) {
        T temp = group.shfl_down(val, offset);
        val = max(val, temp);
    }
    return val;
}

template <typename T>
__device__ T reduce_sum(cg::thread_block_tile<64>& group, T val) {
    for (int offset = group.size()/2; offset > 0; offset /= 2) {
        T temp = group.shfl_down(val, offset);
        val += temp;
    }
    return val;
}

template <typename T>
__device__ T reduce_max(cg::thread_block_tile<64>& group, T val) {
    for (int offset = group.size()/2; offset > 0; offset /= 2) {
        T temp = group.shfl_down(val, offset);
        val = max(val, temp);
    }
    return val;
}

__device__ cg::thread_block this_thread_block() {
    return cg::this_thread_block();
}

template <unsigned Size>
__device__ cg::thread_block_tile<Size> tiled_partition(cg::thread_block& block) {
    return cg::tiled_partition<Size>(block);
}