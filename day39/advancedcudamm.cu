#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
const int K9_NUM_THREADS = 256;

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    sgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr int WM = TM * 16;
    constexpr int WN = TN * 16;
    constexpr int WMITER = CEIL_DIV(BM, WM);
    constexpr int WNITER = CEIL_DIV(BN, WN);

    const int threadCol = threadIdx.x % (WN / TN);
    const int threadRow = threadIdx.x / (WN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

    float threadResults[WMITER * WNITER * TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA)
        {
            float4 tmp = reinterpret_cast<float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB)
        {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<float4 *>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx)
        {
            for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx)
            {
                for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
                {
                    for (uint i = 0; i < TM; ++i)
                    {
                        regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
                    }
                    for (uint i = 0; i < TN; ++i)
                    {
                        regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
                    }
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
                    {
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                        {
                            threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                                          wnIdx * TN + resIdxN] +=
                                regM[resIdxM] * regN[resIdxN];
                        }
                    }
                }
            }
        }
        __syncthreads();
        A += BK;
        B += BK * N;
    }

    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx)
    {
        for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx)
        {
            float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1)
            {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4)
                {
                    float4 tmp = reinterpret_cast<float4 *>(
                        &C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
                                   resIdxN])[0];
                    const int i = (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                    reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}
