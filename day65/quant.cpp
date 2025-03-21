#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <cstdlib>

#define QMIN -128
#define QMAX 127

__global__ void reduceMinMaxKernel(const float* __restrict__ input,
                                   float* __restrict__ partialMins,
                                   float* __restrict__ partialMaxs,
                                   int N, int C)
{
    int channel = blockIdx.y;
    int rowStart = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    float localMin = FLT_MAX;
    float localMax = -FLT_MAX;

    for (int i = rowStart + tid; i < N && i < rowStart + blockDim.x; i += blockDim.x) {
        float val = input[i * C + channel];
        localMin = fminf(localMin, val);
        localMax = fmaxf(localMax, val);
    }

    extern __shared__ float sdata[];
    float* smin = sdata;
    float* smax = sdata + blockDim.x;
    smin[tid] = localMin;
    smax[tid] = localMax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            smin[tid] = fminf(smin[tid], smin[tid + s]);
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int index = channel * gridDim.x + blockIdx.x;
        partialMins[index] = smin[0];
        partialMaxs[index] = smax[0];
    }
}

__global__ void quantizeKernel(const float* __restrict__ input,
                               signed char* __restrict__ output,
                               int totalElements, int C,
                               const float* __restrict__ scales,
                               const float* __restrict__ zeroPoints)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < totalElements) {
        int channel = idx % C;
        float scale = scales[channel];
        float zeroPoint = zeroPoints[channel];
        float q = roundf(input[idx] / scale + zeroPoint);
        q = fminf(fmaxf(q, (float)QMIN), (float)QMAX);
        output[idx] = static_cast<signed char>(q);
    }
}

int main()
{
    const int N = 2048;
    const int C = 32;
    int totalElements = N * C;
    size_t dataSize = totalElements * sizeof(float);
    size_t outputSize = totalElements * sizeof(signed char);

    float* h_input = new float[totalElements];
    signed char* h_output = new signed char[totalElements];
    for (int i = 0; i < totalElements; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }

    float* d_input;
    signed char* d_output;
    hipMalloc(&d_input, dataSize);
    hipMalloc(&d_output, outputSize);
    hipMemcpy(d_input, h_input, dataSize, hipMemcpyHostToDevice);

    int blockSize = 256;
    int numRowBlocks = (N + blockSize - 1) / blockSize;
    dim3 gridReduce(numRowBlocks, C);
    size_t sharedMemSize = 2 * blockSize * sizeof(float);

    int partialSize = C * numRowBlocks;
    float* d_partialMins;
    float* d_partialMaxs;
    hipMalloc(&d_partialMins, partialSize * sizeof(float));
    hipMalloc(&d_partialMaxs, partialSize * sizeof(float));

    hipLaunchKernelGGL(reduceMinMaxKernel,
                       gridReduce,
                       blockSize,
                       sharedMemSize,
                       0,
                       d_input,
                       d_partialMins,
                       d_partialMaxs,
                       N, C);
    hipDeviceSynchronize();

    float* h_partialMins = new float[partialSize];
    float* h_partialMaxs = new float[partialSize];
    hipMemcpy(h_partialMins, d_partialMins, partialSize * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_partialMaxs, d_partialMaxs, partialSize * sizeof(float), hipMemcpyDeviceToHost);

    float* h_scales = new float[C];
    float* h_zeroPoints = new float[C];
    for (int c = 0; c < C; c++) {
        float channelMin = FLT_MAX;
        float channelMax = -FLT_MAX;
        for (int b = 0; b < numRowBlocks; b++) {
            int index = c * numRowBlocks + b;
            channelMin = fminf(channelMin, h_partialMins[index]);
            channelMax = fmaxf(channelMax, h_partialMaxs[index]);
        }
        float scale = (channelMax - channelMin) / (QMAX - QMIN);
        if (scale == 0) scale = 1e-6f;
        float zeroPoint = roundf(QMIN - channelMin / scale);
        zeroPoint = fminf(fmaxf(zeroPoint, (float)QMIN), (float)QMAX);
        h_scales[c] = scale;
        h_zeroPoints[c] = zeroPoint;
        std::cout << "Channel " << c << " | Min: " << channelMin 
                  << " | Max: " << channelMax 
                  << " | Scale: " << scale 
                  << " | ZeroPoint: " << zeroPoint << std::endl;
    }

    float* d_scales;
    float* d_zeroPoints;
    hipMalloc(&d_scales, C * sizeof(float));
    hipMalloc(&d_zeroPoints, C * sizeof(float));
    hipMemcpy(d_scales, h_scales, C * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_zeroPoints, h_zeroPoints, C * sizeof(float), hipMemcpyHostToDevice);

    int totalThreads = totalElements;
    int quantBlockSize = 256;
    int quantNumBlocks = (totalThreads + quantBlockSize - 1) / quantBlockSize;
    hipLaunchKernelGGL(quantizeKernel,
                       quantNumBlocks,
                       quantBlockSize,
                       0, 0,
                       d_input,
                       d_output,
                       totalElements,
                       C,
                       d_scales,
                       d_zeroPoints);
    hipDeviceSynchronize();

    hipMemcpy(h_output, d_output, outputSize, hipMemcpyDeviceToHost);

    std::cout << "\nQuantized Output (first 10 elements):\n";
    for (int i = 0; i < 10; i++) {
        int channel = i % C;
        std::cout << "Input: " << h_input[i] << " (Channel " << channel << ") --> Quantized: "
                  << static_cast<int>(h_output[i]) << " (Scale: " << h_scales[channel]
                  << ", ZeroPoint: " << h_zeroPoints[channel] << ")\n";
    }

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_partialMins);
    hipFree(d_partialMaxs);
    hipFree(d_scales);
    hipFree(d_zeroPoints);
    delete[] h_input;
    delete[] h_output;
    delete[] h_partialMins;
    delete[] h_partialMaxs;
    delete[] h_scales;
    delete[] h_zeroPoints;

    return 0;
}
