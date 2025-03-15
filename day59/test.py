import triton
import triton.language as tl

import torch
import torch.nn as nn
import time

class RMSNorm(nn.Module):
    def __init__(self,dim_embed):
        super().__init__()
        
        self.dim_embed = dim_embed
        self.weight = nn.Parameter(torch.ones((dim_embed),device="cuda"),requires_grad=True)
        
    def forward(self,x):
        rms = torch.sqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + 1e-5)
        x = x / rms
        return x * self.weight
    
@triton.jit
def _rms_kernel(
    X, X_stride,
    O, O_stride,
    W, W_stride,
    eps,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets_col = tl.arange(0, BLOCK_SIZE)
    
    mask = offsets_col < n_cols
    
    X += pid * X_stride
    O += pid * O_stride
    
    X_row = tl.load(X + offsets_col, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + offsets_col, mask=mask, other=0).to(tl.float32)
    
    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_rms = 1.0 / tl.math.sqrt(row_var + eps)  # Corrected: 1/RMS for division
    
    normed = X_row * inv_rms  # This is equivalent to X_row / rms
    output = normed * W_row
    tl.store(O + offsets_col, output, mask=mask)

def triton_rmsnorm(x, weight, eps=1e-5):
    batch, seq_len, hidden_dim = x.shape
    x_reshaped = x.view(-1, hidden_dim)
    output = torch.empty_like(x_reshaped)
    
    x_stride = x_reshaped.stride(0)
    output_stride = output.stride(0)
    weight_stride = 0  
    
    grid = (x_reshaped.shape[0],)
    _rms_kernel[grid](
        x_reshaped.contiguous(), x_stride,
        output, output_stride,
        weight, weight_stride,
        eps,
        hidden_dim,
        BLOCK_SIZE=min(hidden_dim, 1024)  # Use block size up to 1024
    )
    
    # Reshape back to original dimensions
    return output.view(batch, seq_len, hidden_dim)

def test_rmsnorm():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 6
    seq_len = 4096 
    hidden_dim = 2048
    x = torch.randn((batch_size, seq_len, hidden_dim), device=device).to(torch.float32)
    weights = torch.ones((hidden_dim,), device=device).to(torch.float32)
    
    rms_torch = RMSNorm(hidden_dim).to(device)
    rms_torch.weight.data = weights
    
    for _ in range(10):
        output_torch = rms_torch(x)
        output_triton = triton_rmsnorm(x, weights)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        output_torch = rms_torch(x)
    torch.cuda.synchronize()
    torch_time = time.time() - start
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        output_triton = triton_rmsnorm(x, weights)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
    
    print(f"Max difference between PyTorch and Triton: {max_diff:.6f}")
    print(f"PyTorch time: {torch_time:.6f} seconds")
    print(f"Triton time: {triton_time:.6f} seconds")
    print(f"Speedup: {torch_time / triton_time:.2f}x")
    
    return output_torch, output_triton

if __name__ == "__main__":
    torch_result, triton_result = test_rmsnorm()