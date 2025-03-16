import torch
import triton
import triton.language as tl

@triton.jit
def _fused_skip_act_norm_dropout_kernel(
    input_ptr, skip_ptr, output_ptr,
    weight_ptr, bias_ptr,
    M, N,
    stride_input, stride_skip, stride_output,
    dropout_p, seed,
    eps,
    is_training,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offsets = pid * stride_input + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N

    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    skip = tl.load(skip_ptr + offsets, mask=mask, other=0.0)
    
    summed = input + skip

    mean = tl.sum(summed, axis=0) / N
    centered = summed - mean
    var = tl.sum(centered * centered, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)

    normalized = centered * inv_std

    if weight_ptr is not None:
        weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
        normalized *= weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        normalized += bias

    gelu = normalized * 0.5 * (1.0 + tl.erf(normalized / tl.sqrt(2.0)))

    if is_training:
        dropout_mask = tl.rand(seed, offsets) > dropout_p
        gelu = tl.where(dropout_mask, gelu / (1 - dropout_p), 0.0)
    
    tl.store(output_ptr + offsets, gelu, mask=mask)

class FusedSkipNormActDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, skip, weight, bias, p, training, eps):
        assert input.shape == skip.shape
        M, N = input.shape
        output = torch.empty_like(input)
        
        BLOCK_SIZE = triton.next_power_of_2(N)
        
        seed = torch.randint(0, 2**31, (1,)).item()
        
        grid = (M,)
        _fused_skip_act_norm_dropout_kernel[grid](
            input, skip, output,
            weight if weight is not None else None,
            bias if bias is not None else None,
            M, N,
            input.stride(0), skip.stride(0), output.stride(0),
            dropout_p=p,
            seed=seed,
            eps=eps,
            is_training=training,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        ctx.training = training
        ctx.p = p
        ctx.eps = eps
        ctx.save_for_backward(input, skip, weight, bias, output)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward not implemented for this fused operation")

def fused_skip_norm_act_dropout(
    input: torch.Tensor,
    skip: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    p: float = 0.5,
    training: bool = False,
    eps: float = 1e-5
) -> torch.Tensor:
    return FusedSkipNormActDropout.apply(input, skip, weight, bias, p, training, eps)