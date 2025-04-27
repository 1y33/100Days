#include <cuda.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// Block layout: one block per (batch, head)
template<int D>
__global__ void delta_net_attention(
    const half* __restrict__ K,   // [B, S, D]
    const half* __restrict__ V,   // [B, S, D]
    const half* __restrict__ Q,   // [B, S, D]
    half* __restrict__ O,         // [B, S, D]
    int batch, int seq_len)
{
    extern __shared__ half shared_mem[];       // size = D*D
    half* S = shared_mem;                      // state matrix S
    int b = blockIdx.x;                        // batch index

    // Initialize S to zero
    for (int idx = threadIdx.x; idx < D*D; idx += blockDim.x) {
        S[idx] = __float2half(0.0f);
    }
    __syncthreads();

    // Loop over sequence length
    for (int t = 0; t < seq_len; ++t) {
        // Load k_t and v_t into registers
        half k_vec[D], v_vec[D];
        #pragma unroll
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            int base = (b*seq_len + t)*D;
            k_vec[i] = K[base + i];
            v_vec[i] = V[base + i];
        }
        __syncthreads();

        // S += v_vec * k_vec^T  — outer-product update
        for (int i = threadIdx.y; i < D; i += blockDim.y) {
            #pragma unroll
            for (int j = threadIdx.x; j < D; j += blockDim.x) {
                int idx = i*D + j;
                float s = __half2float(S[idx]);
                s += __half2float(v_vec[i]) * __half2float(k_vec[j]);
                S[idx] = __float2half(s);
            }
        }
        __syncthreads();

        // Load q_t and compute o_t = S * q_vec
        half q_vec[D];
        #pragma unroll
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            int base = (b*seq_len + t)*D;
            q_vec[i] = Q[base + i];
        }
        __syncthreads();

        #pragma unroll
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            float o = 0.0f;
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                o += __half2float(S[i*D + j]) * __half2float(q_vec[j]);
            }
            int out_idx = (b*seq_len + t)*D + i;
            O[out_idx] = __float2half(o);
        }
        __syncthreads();
    }
}
