import triton
import triton.language as tl
import torch
import time

@triton.jit
def fused_bias_skip_gelu_scale_kernel(
    x_ptr,
    bias_ptr,
    skip_ptr,
    gamma_ptr,
    y_ptr,
    n_elements: tl.constexpr
):
    BLOCK_SIZE = 1024
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    skip = tl.load(skip_ptr + offsets, mask=mask)
    gamma = tl.load(gamma_ptr + offsets, mask=mask)
    temp = x + bias + skip
    gelu = 0.5 * temp * (1.0 + tl.tanh(0.7978845608028654 * (temp + 0.044715 * temp * temp * temp)))
    out = gelu * gamma
    tl.store(y_ptr + offsets, out, mask=mask)

def test_fused_kernel():
    n_elements = 2048
    BLOCK_SIZE = 1024
    x = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    bias = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    skip = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    gamma = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_bias_skip_gelu_scale_kernel[grid](x, bias, skip, gamma, y, n_elements)
    torch.cuda.synchronize()
    temp = x + bias + skip
    gelu = 0.5 * temp * (1.0 + torch.tanh(0.7978845608028654 * (temp + 0.044715 * temp ** 3)))
    ref = gelu * gamma
    if torch.allclose(y, ref, atol=1e-6):
        print("Test passed! Kernel output matches reference.")
    else:
        print("Test failed! Maximum absolute error:", (y - ref).abs().max().item())

def benchmark_kernel():
    n_elements = 2048
    BLOCK_SIZE = 1024
    x = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    bias = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    skip = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    gamma = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    for _ in range(10):
        fused_bias_skip_gelu_scale_kernel[grid](x, bias, skip, gamma, y, n_elements)
    torch.cuda.synchronize()
    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        fused_bias_skip_gelu_scale_kernel[grid](x, bias, skip, gamma, y, n_elements)
    torch.cuda.synchronize()
    end = time.time()
    avg_time_ms = (end - start) / n_iter * 1000
    print(f"Average kernel time over {n_iter} iterations: {avg_time_ms:.3f} ms")

if __name__ == "__main__":
    test_fused_kernel()
    benchmark_kernel()
