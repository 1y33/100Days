#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

__device__ inline float gelu(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    float t = tanhf(k0 * (x + k1 * x * x * x));
    return 0.5f * x * (1.0f + t);
}

__device__ inline float gelu_grad(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    float t_arg = k0 * (x + k1 * x * x * x);
    float t = tanhf(t_arg);
    float dt = (1.0f - t * t);
    float left = 0.5f * (1.0f + t);
    float right = 0.5f * x * k0 * dt * (1.0f + 3.0f * k1 * x * x);
    return left + right;
}

extern "C" __global__ void geglu_ffn_forward(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_features,
    int hidden_features)
{
    int b = blockIdx.x;
    int h = threadIdx.x;

    if (b < batch && h < hidden_features) {
        float a_val = 0.0f;
        float b_val = 0.0f;
        for (int i = 0; i < in_features; i++) {
            float x = input[b * in_features + i];
            float w_a = weight[i * (2 * hidden_features) + h];
            float w_b = weight[i * (2 * hidden_features) + h + hidden_features];
            a_val += x * w_a;
            b_val += x * w_b;
        }
        if (bias != nullptr) {
            a_val += bias[h];
            b_val += bias[h + hidden_features];
        }
        output[b * hidden_features + h] = a_val * gelu(b_val);
    }
}

extern "C" __global__ void geglu_ffn_backward(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int batch,
    int in_features,
    int hidden_features)
{
    int b_idx = blockIdx.x;
    int h = threadIdx.x;

    if (b_idx < batch && h < hidden_features) {
        float a_val = 0.0f;
        float b_val = 0.0f;
        for (int i = 0; i < in_features; i++) {
            float x = input[b_idx * in_features + i];
            a_val += x * weight[i * (2 * hidden_features) + h];
            b_val += x * weight[i * (2 * hidden_features) + h + hidden_features];
        }
        if (bias != nullptr) {
            a_val += bias[h];
            b_val += bias[h + hidden_features];
        }
        float dY = grad_output[b_idx * hidden_features + h];
        float grad_a = dY * gelu(b_val);
        float grad_b = dY * a_val * gelu_grad(b_val);

        for (int i = 0; i < in_features; i++) {
            float x = input[b_idx * in_features + i];
            atomicAdd(&grad_input[b_idx * in_features + i],
                      grad_a * weight[i * (2 * hidden_features) + h] +
                      grad_b * weight[i * (2 * hidden_features) + h + hidden_features]);

            atomicAdd(&grad_weight[i * (2 * hidden_features) + h], x * grad_a);
            atomicAdd(&grad_weight[i * (2 * hidden_features) + h + hidden_features], x * grad_b);
        }
        if (grad_bias != nullptr) {
            atomicAdd(&grad_bias[h], grad_a);
            atomicAdd(&grad_bias[h + hidden_features], grad_b);
        }
    }
}

void launch_geglu_ffn_forward(const float* input, const float* weight, const float* bias,
                              float* output, int batch, int in_features, int hidden_features)
{
    dim3 grid(batch);
    dim3 block(hidden_features);
    hipLaunchKernelGGL(geglu_ffn_forward, grid, block, 0, 0,
                       input, weight, bias, output, batch, in_features, hidden_features);
}

void launch_geglu_ffn_backward(const float* input, const float* weight, const float* bias,
                               const float* grad_output, float* grad_input,
                               float* grad_weight, float* grad_bias,
                               int batch, int in_features, int hidden_features)
{
    dim3 grid(batch);
    dim3 block(hidden_features);
    hipLaunchKernelGGL(geglu_ffn_backward, grid, block, 0, 0,
                       input, weight, bias, grad_output,
                       grad_input, grad_weight, grad_bias,
                       batch, in_features, hidden_features);
}

// int main() {
//     return 0;
// }
