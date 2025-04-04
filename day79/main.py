import torch
import triton
import triton.language as tl

@triton.jit
def quantize_kernel(input_ptr, output_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    x_scaled = x * scale
    
    x_rounded = tl.round(x_scaled)
    
    x_clamped = tl.max(tl.min(x_rounded, 127), -128)
    
    tl.store(output_ptr + offsets, tl.cast(x_clamped, tl.int8), mask=mask)

def quantize(input_tensor, scale):
    
    assert input_tensor.is_cuda, "Input tensor must be on a CUDA device"
    n_elements = input_tensor.numel()
    output_tensor = torch.empty_like(input_tensor, dtype=torch.int8)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    quantize_kernel[grid](input_tensor, output_tensor, n_elements, scale, BLOCK_SIZE=1024)
    
    return output_tensor

if __name__ == '__main__':
    
    input_tensor = torch.randn(1024 * 1024, device='cuda', dtype=torch.float32)
    scale = 127.0
    output_tensor = quantize(input_tensor, scale)
    print("Quantization complete. Output tensor:")
    print(output_tensor)
