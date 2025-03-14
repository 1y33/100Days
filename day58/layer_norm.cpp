#include <hip/hip_runtime.h>

__global__ void layer_norm_kernel(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    int batch_size,
    int hidden_size,
    float epsilon)
{
    extern __shared__ float shared[];
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    float* sum = shared;
    float* sum_sq = &shared[blockDim.x];

    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = input[batch_idx * hidden_size + i];
        thread_sum += val;
    }
    sum[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum[tid] += sum[tid + stride];
        }
        __syncthreads();
    }
    float mean = sum[0] / hidden_size;

    float thread_sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = input[batch_idx * hidden_size + i];
        float diff = val - mean;
        thread_sum_sq += diff * diff;
    }
    sum_sq[tid] = thread_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_sq[tid] += sum_sq[tid + stride];
        }
        __syncthreads();
    }
    float variance = sum_sq[0] / hidden_size + epsilon;
    float inv_std = rsqrtf(variance);

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = input[batch_idx * hidden_size + i];
        float normalized = (val - mean) * inv_std;
        output[batch_idx * hidden_size + i] = normalized * gamma[i] + beta[i];
    }
}

void layer_norm_hip(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    int batch_size,
    int hidden_size,
    float epsilon,
    hipStream_t stream)
{
    dim3 blocks(batch_size);
    dim3 threads(256);
    size_t shared_mem = 2 * threads.x * sizeof(float);

    hipLaunchKernelGGL(
        layer_norm_kernel,
        blocks, threads, shared_mem, stream,
        output, input, gamma, beta,
        batch_size, hidden_size, epsilon
    );
}