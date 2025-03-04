import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    coeff1 = 0.7978845608028654
    coeff2 = 0.044715
    x_cubed = x * x * x
    inner = coeff1 * (x + coeff2 * x_cubed)
    tanh = tl.math.tanh(inner)
    output = 0.5 * x * (1.0 + tanh)
    
    tl.store(output_ptr + offsets, output, mask=mask)

def fused_gelu(x: torch.Tensor):
    output = torch.empty_like(x)
    
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    
    x = torch.randn(1000000, device='cuda')
    
    triton_output = fused_gelu(x)
    
    torch_output = torch.nn.functional.gelu(x)
    
    print(f"Maximum absolute error: {torch.max(torch.abs(triton_output - torch_output)):.2e}")
    print(f"Results match: {torch.allclose(triton_output, torch_output, atol=1e-5)}")