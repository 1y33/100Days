import torch
import triton
import triton.language as tl

def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

@triton.jit
def _rms_norm_fwd_fused(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_start_ptr = X_ptr + row_idx * stride_x_row
    y_row_start_ptr = Y_ptr + row_idx * stride_y_row

    row_var = tl.zeros((), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_SIZE_N):
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask = offs_n < N
        x = tl.load(x_row_start_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
        row_var += tl.sum(x * x, axis=0)

    variance = row_var / N
    variance_eps = variance + eps
    rstd = tl.math.rsqrt(variance_eps)

    for start_n in range(0, N, BLOCK_SIZE_N):
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask = offs_n < N
        x = tl.load(x_row_start_ptr + offs_n, mask=mask, other=0.0)
        w = tl.load(W_ptr + offs_n, mask=mask)
        x_norm = x * rstd.to(x.dtype)
        y = x_norm * w
        tl.store(y_row_start_ptr + offs_n, y, mask=mask)

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda, "Weight tensor must be on CUDA"
    assert x.shape[-1] == weight.shape[-1], f"Input feature dim ({x.shape[-1]}) must match weight dim ({weight.shape[-1]})"
    assert weight.ndim == 1, "Weight tensor must be 1-dimensional"
    assert x.is_contiguous() or x.stride(-1) == 1, "Input tensor's last dimension must be contiguous"
    assert weight.is_contiguous(), "Weight tensor must be contiguous"

    input_shape = x.shape
    M, N = x.numel() // input_shape[-1], input_shape[-1]
    x_2d = x.view(M, N)
    y = torch.empty_like(x_2d)

    BLOCK_SIZE_N = max(16, min(triton.next_power_of_2(N), 4096))
    grid = (M,)

    _rms_norm_fwd_fused[grid](
        x_2d,
        y,
        weight,
        x_2d.stride(0),
        y.stride(0),
        N,
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return y.view(input_shape)

if __name__ == "__main__":
    dtype = torch.float16
    device = 'cuda'

    batch_size = 4
    seq_len = 512
    hidden_dim = 1024

    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device, requires_grad=False)
    weight = torch.rand(hidden_dim, dtype=dtype, device=device, requires_grad=False) * 2 + 0.5

    x = x.contiguous()
    weight = weight.contiguous()

    epsilon = 1e-5

    print("Running Triton RMSNorm...")
    y_triton = rms_norm(x, weight, epsilon)
    print("Triton RMSNorm finished.")

    print("Running PyTorch RMSNorm for comparison...")
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normalized_pytorch = x * torch.rsqrt(variance + epsilon)
    y_pytorch = x_normalized_pytorch * weight.to(x_normalized_pytorch.dtype)
    print("PyTorch RMSNorm finished.")

    print("Comparing outputs...")
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    rtol = 1e-2 if dtype == torch.float16 else 1e-4
    try:
        torch.allclose(y_triton, y_pytorch, atol=atol, rtol=rtol)
        print("Verification Successful: Triton and PyTorch outputs match.")
    except RuntimeError as e:
        print(f"Verification FAILED: {e}")
        print("Max difference:", torch.max(torch.abs(y_triton - y_pytorch)))

    print("\nBenchmarking...")

    def run_triton():
        rms_norm(x, weight, epsilon)
        torch.cuda.synchronize()

    def run_pytorch():
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normalized_pytorch = x * torch.rsqrt(variance + epsilon)
        y_pytorch = x_normalized_pytorch * weight.to(x_normalized_pytorch.dtype)
        torch.cuda.synchronize()

    for _ in range(10):
        run_triton()
        run_pytorch()

    ms_triton = triton.testing.do_bench(run_triton, rep=100)
    ms_pytorch = triton.testing.do_bench(run_pytorch, rep=100)

    print(f"Triton RMSNorm Forward: {ms_triton:.4f} ms")
    print(f"PyTorch RMSNorm Forward: {ms_pytorch:.4f} ms")
    print(f"Speedup: {ms_pytorch / ms_triton:.2f}x")
