import triton
import triton.language as tl
import torch
import time

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 128, 'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_T': 32, 'BLOCK_D': 256}, num_warps=8),
    ],
    key=['seq_len', 'd_model'],
)
@triton.jit
def rotary_kernel_optimized(
    x_ptr, cos_ptr, sin_ptr, y_ptr,
    batch, seq_len, n_heads, d_model,
    seq_offset: tl.constexpr,
    conjugate: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D//2)
    positions = (offs_t % seq_len) + seq_offset
    batch_idx = offs_t // seq_len
    t_mask = offs_t < (batch * seq_len)
    d_mask = offs_d < (d_model//2)
    mask_2d = t_mask[:, None] & d_mask[None, :]
    pos_seq = positions % seq_len
    cos_sin_idx = pos_seq[:, None] * (d_model//2) + offs_d[None, :]
    pos_mask = pos_seq[:, None] < seq_len
    load_mask = mask_2d & pos_mask
    cos_values = tl.load(cos_ptr + cos_sin_idx, mask=load_mask, other=0.0)
    sin_values = tl.load(sin_ptr + cos_sin_idx, mask=load_mask, other=0.0)
    if conjugate:
        sin_values = -sin_values
    x_idx_base = offs_t[:, None]
    for h in range(n_heads):
        x0_idx = x_idx_base * (n_heads * d_model) + h * d_model + offs_d[None, :]
        x1_idx = x_idx_base * (n_heads * d_model) + h * d_model + (d_model//2 + offs_d[None, :])
        x0 = tl.load(x_ptr + x0_idx, mask=mask_2d, other=0.0)
        x1 = tl.load(x_ptr + x1_idx, mask=mask_2d, other=0.0)
        y0 = x0 * cos_values - x1 * sin_values
        y1 = x0 * sin_values + x1 * cos_values
        tl.store(y_ptr + x0_idx, y0, mask=mask_2d)
        tl.store(y_ptr + x1_idx, y1, mask=mask_2d)

def apply_rotary_optimized(x, cos, sin, seq_offset=0):
    batch, seq_len, n_heads, d_model = x.shape
    x_reshaped = x.reshape(batch * seq_len, n_heads * d_model)
    y_reshaped = torch.empty_like(x_reshaped)
    grid = lambda meta: (
        triton.cdiv(batch * seq_len, meta['BLOCK_T']),
        triton.cdiv(d_model//2, meta['BLOCK_D']),
    )
    kwargs = {
        'x_ptr': x_reshaped,
        'cos_ptr': cos,
        'sin_ptr': sin,
        'y_ptr': y_reshaped,
        'batch': batch,
        'seq_len': seq_len,
        'n_heads': n_heads,
        'd_model': d_model,
        'seq_offset': seq_offset,
        'conjugate': False
    }
    rotary_kernel_optimized[grid](**kwargs)
    return y_reshaped.reshape(batch, seq_len, n_heads, d_model)

def compare_speed_all():
    batch, seq_len, n_heads, d_model = 32, 128, 12, 64
    x = torch.randn(batch, seq_len, n_heads, d_model, device='cuda')
    cos = torch.randn(seq_len, d_model//2, device='cuda')
    sin = torch.randn(seq_len, d_model//2, device='cuda')
    for _ in range(10):
        _ = apply_rotary(x, cos, sin)
        _ = apply_rotary_optimized(x, cos, sin)
        _ = apply_rotary_pytorch(x, cos, sin)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y_triton = apply_rotary(x, cos, sin)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 100
    start = time.time()
    for _ in range(100):
        y_optimized = apply_rotary_optimized(x, cos, sin)
    torch.cuda.synchronize()
    optimized_time = (time.time() - start) / 100
    start = time.time()
    for _ in range(100):
        y_pytorch = apply_rotary_pytorch(x, cos, sin)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 100
    torch.testing.assert_close(y_triton, y_pytorch, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(y_optimized, y_pytorch, rtol=1e-2, atol=1e-2)
    print(f"Original Triton implementation time: {triton_time*1000:.3f} ms")
    print(f"Optimized Triton implementation time: {optimized_time*1000:.3f} ms")
    print(f"PyTorch implementation time: {pytorch_time*1000:.3f} ms")
    print(f"Speedup over original Triton: {triton_time/optimized_time:.2f}x")
    print(f"Speedup over PyTorch: {pytorch_time/optimized_time:.2f}x")

if __name__ == "__main__":
    compare_speed_all()
