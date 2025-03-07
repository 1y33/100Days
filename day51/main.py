import torch
import torch.nn as nn
import triton
import triton.language as tl
import matplotlib.pyplot as plt

@triton.jit
def quantize_asymmetric_kernel(
    x_ptr,          # input tensor
    q_ptr,          #  output tensor
    scale_ptr,      # pointer to scale value (1 per block)
    zero_ptr,       # pointer to zero point value (1 per block)
    bits,           # number of bits to quantize to
    n_elements,     # number of elements in tensor
    BLOCK_SIZE: tl.constexpr,  # block size for processing
):
    pid = tl.program_id(0)
    block_offset = pid * BLOCK_SIZE
    
    offset = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask)
    
    alpha = tl.max(x)  
    beta = tl.min(x)   
    
    scale = (alpha - beta) / (tl.exp2(bits) - 1)
    zero_point = tl.zeros_like(scale)
    
    has_range = alpha > beta
    if has_range:
        zero_point = round(-beta / scale)
    
    lower_bound, upper_bound = 0, tl.exp2(bits) - 1
    q = tl.clamp(round(x / scale + zero_point), lower_bound, upper_bound)
    
    tl.store(q_ptr + offset, q, mask=mask)
    tl.store(scale_ptr, scale)
    tl.store(zero_ptr, zero_point)

@triton.jit
def dequantize_asymmetric_kernel(
    q_ptr,  # pointer to quantized tensor
    x_ptr,  # pointer to output dequantized tensor
    scale_ptr,  # pointer to scale value
    zero_ptr,  # pointer to zero point value
    n_elements,  # number of elements in tensor
    BLOCK_SIZE: tl.constexpr,  # block size for processing
):
    # Program ID and block offset
    pid = tl.program_id(0)
    block_offset = pid * BLOCK_SIZE
    
    # Create mask for valid elements
    offset = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load quantized data, scale and zero point
    q = tl.load(q_ptr + offset, mask=mask)
    scale = tl.load(scale_ptr)
    zero_point = tl.load(zero_ptr)
    
    x = scale * (q - zero_point)
    tl.store(x_ptr + offset, x, mask=mask)

def quantize_asymmetric(x, bits=8):
    n_elements = x.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    
    q = torch.empty_like(x, dtype=torch.float32)
    scale = torch.zeros(1, device=x.device, dtype=torch.float32)
    zero_point = torch.zeros(1, device=x.device, dtype=torch.float32)
    
    quantize_asymmetric_kernel[grid](
        x_ptr=x, 
        q_ptr=q, 
        scale_ptr=scale, 
        zero_ptr=zero_point, 
        bits=bits,
        n_elements=n_elements, 
        BLOCK_SIZE=block_size
    )
    
    return q, scale, zero_point

def dequantize_asymmetric(q, scale, zero_point):
    n_elements = q.numel()
    block_size = 1024  
    grid = (triton.cdiv(n_elements, block_size),)
    
    x = torch.empty_like(q, dtype=torch.float32)
    
    dequantize_asymmetric_kernel[grid](
        q_ptr=q, 
        x_ptr=x, 
        scale_ptr=scale, 
        zero_ptr=zero_point,
        n_elements=n_elements, 
        BLOCK_SIZE=block_size
    )
    
    return x

def main():
    # Generate random tensor to simulate weights
    torch.manual_seed(42)
    original_weights = torch.randn(1024, device='cuda')
    
    print(f"Original weights shape: {original_weights.shape}")
    print(f"Original weights range: [{original_weights.min().item():.4f}, {original_weights.max().item():.4f}]")
    
    # Quantize with different bit widths
    bit_widths = [2, 4, 8]
    
    plt.figure(figsize=(15, 10))
    plt.subplot(len(bit_widths) + 1, 1, 1)
    plt.hist(original_weights.cpu().numpy(), bins=50, alpha=0.7)
    plt.title("Original Weights Distribution")
    plt.grid(True)
    
    for i, bits in enumerate(bit_widths):
        quantized, scale, zero_point = quantize_asymmetric(original_weights, bits)
        
        dequantized = dequantize_asymmetric(quantized, scale, zero_point)
        
        error = torch.abs(original_weights - dequantized).mean().item()
        
        print(f"\n{bits}-bit Quantization:")
        print(f"  Scale: {scale.item():.6f}, Zero Point: {zero_point.item():.2f}")
        print(f"  Quantized range: [{quantized.min().item():.1f}, {quantized.max().item():.1f}]")
        print(f"  Mean Absolute Error: {error:.6f}")
        
        plt.subplot(len(bit_widths) + 1, 1, i + 2)
        plt.hist(quantized.cpu().numpy(), bins=2**bits, alpha=0.7)
        plt.title(f"{bits}-bit Quantized Weights (MAE: {error:.6f})")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("quantization_comparison.png")
    print("\nVisualization saved to 'quantization_comparison.png'")

if __name__ == "__main__":
    main()