import torch
import triton
import triton.language as tl

@triton.jit
def lora_kernel(
    y_ptr, x_ptr, w_ptr, a_ptr, b_ptr,
    M, N, K, R,
    stride_ym, stride_yn,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ak, stride_ar,
    stride_br, stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_r = tl.arange(0, BLOCK_SIZE_R)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask_m = (offs_m < M)[:, None]
    mask_n = (offs_n < N)[None, :]
    mask_y = mask_m & mask_n

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk
        mask_x = (offs_m < M)[:, None] & ((k + offs_k) < K)[None, :]
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_ptr + (k + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn
        mask_w = ((k + offs_k) < K)[:, None] & (offs_n < N)[None, :]
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        ab = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        for r in range(0, R, BLOCK_SIZE_R):
            a_ptrs = a_ptr + (k + offs_k)[:, None] * stride_ak + (r + offs_r)[None, :] * stride_ar
            mask_a = ((k + offs_k) < K)[:, None] & ((r + offs_r) < R)[None, :]
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)

            b_ptrs = b_ptr + (r + offs_r)[:, None] * stride_br + offs_n[None, :] * stride_bn
            mask_b = ((r + offs_r) < R)[:, None] & (offs_n < N)[None, :]
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

            ab += tl.dot(a.to(tl.float32), b.to(tl.float32))

        w_eff = w.to(tl.float32) + ab
        acc += tl.dot(x.to(tl.float32), w_eff)

    tl.store(y_ptrs, acc.to(tl.float16), mask=mask_y)

def lora_matmul(x, W, A, B):
    M, K = x.shape
    _, N = W.shape
    R = A.shape[1]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_R = 32
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    
    lora_kernel[grid](
        y, x, W, A, B,
        M, N, K, R,
        y.stride(0), y.stride(1),
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_R=BLOCK_SIZE_R,
    )
    return y
