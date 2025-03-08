import triton
import numpy as np

import triton.language as tl

@triton.jit
def init_matrix(matrix, seed: tl.constexpr):
    idx = tl.arange(0, matrix.shape[0])
    matrix[idx] = tl.random(seed + idx)

@triton.jit
def add_matrices(a, b, result):
    idx = tl.arange(0, a.shape[0])
    result[idx] = a[idx] + b[idx]

@triton.jit
def multiply_matrices(a, b, result):
    idx = tl.arange(0, a.shape[0])
    result[idx] = a[idx] * b[idx]

@triton.jit
def transpose_matrix(matrix, result):
    idx = tl.arange(0, matrix.shape[0])
    idy = tl.arange(0, matrix.shape[1])
    result[idy, idx] = matrix[idx, idy]

@triton.jit
def matmul_kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // N
    col = pid % N

    acc = 0.0
    for k in range(K):
        acc += a[row, k] * b[k, col]

    c[row, col] = acc

if __name__ == "__main__":

    M, N, K = 128, 128, 128
    a = np.random.rand(M, K).astype(np.float32)
    b = np.random.rand(K, N).astype(np.float32)
    c = np.zeros((M, N), dtype=np.float32)

    a_dev = triton.testing.to_device(a)
    b_dev = triton.testing.to_device(b)
    c_dev = triton.testing.to_device(c)

    grid = (M * N,)
    matmul_kernel[grid](a_dev, b_dev, c_dev, M, N, K)

    c = c_dev.cpu()
    print(c)