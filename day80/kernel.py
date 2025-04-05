import torch
import triton
import triton.language as tl

@triton.jit
def rwkv_kernel(
    output_ptr,    
    k_ptr,        
    v_ptr,         
    w_ptr,         
    n_time: tl.constexpr,      
    n_channels: tl.constexpr,  
    stride_time: tl.constexpr, 
    stride_batch: tl.constexpr 
):
    pid = tl.program_id(0)
    batch = pid // n_channels   
    channel = pid % n_channels 

    w = tl.load(w_ptr + channel)

    max_val = -1e30
    numerator = 0.0
    denominator = 0.0

    for t in range(n_time):
        offset = batch * stride_batch + t * stride_time + channel

        cur_k = tl.load(k_ptr + offset)
        cur_v = tl.load(v_ptr + offset)

        m = tl.maximum(max_val, cur_k)

        exp_max_diff = tl.exp(max_val - m)
        exp_k_diff = tl.exp(cur_k - m)

        numerator = numerator * exp_max_diff + cur_v * exp_k_diff
        denominator = denominator * exp_max_diff + exp_k_diff

        result = numerator / denominator
        tl.store(output_ptr + offset, result)

        max_val = m + w

def rwkv_forward(k: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    assert k.is_cuda and v.is_cuda and w.is_cuda, "All tensors must be on CUDA."
    B, T, C = k.shape

    output = torch.empty_like(v)

    stride_time = k.stride(1)
    stride_batch = k.stride(0)

    grid = (B * C,)

    rwkv_kernel[grid](
        output_ptr=output,
        k_ptr=k,
        v_ptr=v,
        w_ptr=w,
        n_time=T,
        n_channels=C,
        stride_time=stride_time,
        stride_batch=stride_batch,
    )
    return output

if __name__ == '__main__':
    B = 2           # batch size
    T = 128         # sequence length
    C = 256         # number of channels

    k_tensor = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    v_tensor = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    w_tensor = torch.randn(C, device='cuda', dtype=torch.float32) * 0.1

    output_tensor = rwkv_forward(k_tensor, v_tensor, w_tensor)
    print("Output shape:", output_tensor.shape)
    print("Output sample:", output_tensor[0, :5, :5])
