import triton
import triton.language as tl
import torch

@triton.jit
def fp8_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale_a, scale_b, scale_c,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    offm = pid_m * BLOCK_M + rm
    offn = pid_n * BLOCK_N + rn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offk = k + tl.arange(0, BLOCK_K)

        a = tl.load(
            a_ptr + offm[:, None] * stride_am + offk[None, :] * stride_ak,
            mask=(offm[:, None] < M) & (offk[None, :] < K),
            other=0,
        )
        b = tl.load(
            b_ptr + offk[:, None] * stride_bk + offn[None, :] * stride_bn,
            mask=(offk[:, None] < K) & (offn[None, :] < N),
            other=0,
        )

        a_fp32 = tl.cast(a, tl.float32) * scale_a
        b_fp32 = tl.cast(b, tl.float32) * scale_b

        acc += tl.dot(a_fp32, b_fp32)

    c_fp8 = tl.round(acc / scale_c)
    c_fp8 = tl.max(tl.min(c_fp8, 127), -128)
    
    tl.store(
        c_ptr + offm[:, None] * stride_cm + offn[None, :] * stride_cn,
        c_fp8.to(tl.int8),
        mask=(offm[:, None] < M) & (offn[None, :] < N)
    )

def fp8_gemm(a: torch.Tensor, b: torch.Tensor,
             scale_a: float, scale_b: float, scale_c: float,
             BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32) -> torch.Tensor:
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    
    c = torch.empty((M, N), device=a.device, dtype=torch.int8)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    fp8_gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scale_a, scale_b, scale_c,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return c

if __name__ == "__main__":
    torch.manual_seed(0)
    M, K, N = 128, 256, 64

    a_fp8 = torch.randint(-128, 127, (M, K), device='cuda', dtype=torch.int8)
    b_fp8 = torch.randint(-128, 127, (K, N), device='cuda', dtype=torch.int8)
    
    scale_a, scale_b, scale_c = 0.1, 0.1, 0.05
    
    c_fp8 = fp8_gemm(a_fp8, b_fp8, scale_a, scale_b, scale_c)
    print("GEMM result (FP8 stored as int8):", c_fp8)
