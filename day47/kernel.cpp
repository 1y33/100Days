#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;
using namespace cooperative_groups;

constexpr int WARPS_PER_BLOCK = 4;
constexpr int WAVE_SIZE = 64;
constexpr int HEAD_DIM = 64;
constexpr int TILE_DIM = 16;
constexpr int MFMA_GROUP_SIZE = 16;

__global__ void fused_multi_head_attention_kernel(
    __half2* __restrict__ output,
    const __half2* __restrict__ queries,
    const __half2* __restrict__ keys,
    const __half2* __restrict__ values,
    int batch_size,
    int seq_len,
    int num_heads,
    float scale) {
    
    extern __shared__ __half2 shared_mem[];
    thread_block blk = this_thread_block();
    thread_block_tile<MFMA_GROUP_SIZE> wave = tiled_partition<MFMA_GROUP_SIZE>(blk);

    const int head_id = blockIdx.y;
    const int batch_id = blockIdx.z;
    const int wave_id = threadIdx.x / WAVE_SIZE;
    const int lane_id = threadIdx.x % WAVE_SIZE;

    fragment<matrix_a, MFMA_GROUP_SIZE, MFMA_GROUP_SIZE, TILE_DIM, __half, row_major> q_frag;
    fragment<matrix_b, MFMA_GROUP_SIZE, MFMA_GROUP_SIZE, TILE_DIM, __half, col_major> k_frag;
    fragment<accumulator, MFMA_GROUP_SIZE, MFMA_GROUP_SIZE, TILE_DIM, float> acc_frag;
    
    fill_fragment(acc_frag, 0.0f);

    __half2* q_shared = shared_mem;
    __half2* k_shared = shared_mem + (TILE_DIM * HEAD_DIM / 2) * WARPS_PER_BLOCK;

    const int q_offset = batch_id * num_heads * seq_len * HEAD_DIM + 
                       head_id * seq_len * HEAD_DIM;
    const int kv_offset = batch_id * num_heads * seq_len * HEAD_DIM + 
                        head_id * seq_len * HEAD_DIM;

    for(int tile = 0; tile < seq_len; tile += TILE_DIM) {
        for(int i = wave_id; i < TILE_DIM; i += WARPS_PER_BLOCK) {
            for(int j = lane_id; j < HEAD_DIM/2; j += WAVE_SIZE) {
                q_shared[i * HEAD_DIM/2 + j] = queries[q_offset + (tile + i) * HEAD_DIM/2 + j];
            }
        }

        for(int i = wave_id; i < TILE_DIM; i += WARPS_PER_BLOCK) {
            for(int j = lane_id; j < HEAD_DIM/2; j += WAVE_SIZE) {
                k_shared[i * HEAD_DIM/2 + j] = keys[kv_offset + (tile + i) * HEAD_DIM/2 + j];
            }
        }
        
        blk.sync();

        for(int i = 0; i < TILE_DIM; i += MFMA_GROUP_SIZE) {
            load_matrix_sync(q_frag, (__half*)(q_shared + i * HEAD_DIM/2), HEAD_DIM);
            load_matrix_sync(k_frag, (__half*)(k_shared + i * HEAD_DIM/2), HEAD_DIM);
            mma_sync(acc_frag, q_frag, k_frag, acc_frag);
        }
    }

    float max_val = -INFINITY;
    float sum = 0.0f;

    for(int i = lane_id; i < seq_len; i += WAVE_SIZE) {
        max_val = fmax(max_val, acc_frag.x[i]);
    }
    max_val = reduce_max(max_val);

    for(int i = lane_id; i < seq_len; i += WAVE_SIZE) {
        float val = expf((acc_frag.x[i] - max_val) * scale);
        sum += val;
        acc_frag.x[i] = val;
    }
    sum = reduce_sum(sum);

    for(int i = lane_id; i < seq_len; i += WAVE_SIZE) {
        acc_frag.x[i] /= sum;
    }

    const int out_offset = batch_id * num_heads * seq_len * HEAD_DIM + 
                         head_id * seq_len * HEAD_DIM;

    for(int i = wave_id; i < TILE_DIM; i += WARPS_PER_BLOCK) {
        for(int j = lane_id; j < HEAD_DIM/2; j += WAVE_SIZE) {
            output[out_offset + i * HEAD_DIM/2 + j] = 
                __hfma2(__float2half2_rn(acc_frag.x[i * TILE_DIM + j]), 
                       ((__half2*)values)[out_offset + i * HEAD_DIM/2 + j],
                       ((__half2*)output)[out_offset + i * HEAD_DIM/2 + j]);
        }
    }
}

void multi_gpu_attention_forward(
    __half2* local_output,
    __half2* local_queries,
    __half2* remote_keys,
    __half2* remote_values,
    int batch_size,
    int seq_len,
    int num_heads,
    int current_gpu,
    int peer_gpu_id) {
    
    hipSetDevice(current_gpu);
    hipError_t status = hipDeviceEnablePeerAccess(peer_gpu_id, 0);
    if(status != hipSuccess) {
        std::cerr << "Failed to enable peer access: " << hipGetErrorString(status) << std::endl;
    }

    __half2 *keys_staging, *values_staging;
    hipMalloc(&keys_staging, batch_size * num_heads * seq_len * HEAD_DIM * sizeof(__half2));
    hipMalloc(&values_staging, batch_size * num_heads * seq_len * HEAD_DIM * sizeof(__half2));

    hipStream_t copy_stream, compute_stream;
    hipStreamCreate(&copy_stream);
    hipStreamCreate(&compute_stream);

    hipMemcpyPeerAsync(keys_staging, current_gpu,
                      remote_keys, peer_gpu_id,
                      batch_size * num_heads * seq_len * HEAD_DIM * sizeof(__half2),
                      copy_stream);

    dim3 grid((seq_len + TILE_DIM - 1)/TILE_DIM, num_heads, batch_size);
    dim3 block(WAVE_SIZE * WARPS_PER_BLOCK);
    size_t shared_mem = (TILE_DIM * HEAD_DIM * 2) * sizeof(__half2);

    hipLaunchKernelGGL(fused_multi_head_attention_kernel, 
                      grid, block, shared_mem, compute_stream,
                      local_output, local_queries, keys_staging, values_staging,
                      batch_size, seq_len, num_heads, 1.0f/sqrtf(HEAD_DIM));

    hipMemcpyPeerAsync(values_staging, current_gpu,
                      remote_values, peer_gpu_id,
                      batch_size * num_heads * seq_len * HEAD_DIM * sizeof(__half2),
                      copy_stream);

    hipStreamDestroy(copy_stream);
    hipStreamDestroy(compute_stream);
    hipFree(keys_staging);
    hipFree(values_staging);
}

int main() {
    int batch_size = 8;
    int seq_len = 1024;
    int num_heads = 12;
    int current_gpu = 0;
    int peer_gpu_id = 1;

    hipSetDevice(current_gpu);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, current_gpu);
    std::cout << "Running on: " << prop.name << std::endl;

    __half2 *local_output, *local_queries, *remote_keys, *remote_values;
    hipMalloc(&local_output, batch_size * num_heads * seq_len * HEAD_DIM * sizeof(__half2));
    hipMalloc(&local_queries, batch_size * num_heads * seq_len * HEAD_DIM * sizeof(__half2));

    hipSetDevice(peer_gpu_id);
    hipMalloc(&remote_keys, batch_size * num_heads * seq_len * HEAD_DIM * sizeof(__half2));
    hipMalloc(&remote_values, batch_size * num_heads * seq_len * HEAD_DIM * sizeof(__half2));

    multi_gpu_attention_forward(local_output, local_queries, remote_keys, remote_values,
                               batch_size, seq_len, num_heads, current_gpu, peer_gpu_id);

    hipDeviceSynchronize();
    hipFree(local_output);
    hipFree(local_queries);
    hipFree(remote_keys);
    hipFree(remote_values);

    return 0;
}