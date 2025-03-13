import triton
import triton.language as tl
import torch
@triton.jit
def fused_linear_xentropy_forward(
    input_ptr, weight_ptr, bias_ptr, target_ptr, loss_ptr,
    batch_size, in_features, out_features,
    stride_input_batch, stride_input_feature,
    stride_weight_out, stride_weight_in,
    stride_bias_out,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    input_row = input_ptr + pid * stride_input_batch
    target = tl.load(target_ptr + pid)

    logits = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)
    
    for i in range(0, in_features, BLOCK_SIZE_IN):
        input_offsets = i + tl.arange(0, BLOCK_SIZE_IN)
        input_mask = input_offsets < in_features
        current_input = tl.load(input_row + input_offsets, mask=input_mask, other=0.0)

        weight_offsets = (i + tl.arange(0, BLOCK_SIZE_IN))[None, :] * stride_weight_in + \
                        tl.arange(0, BLOCK_SIZE_OUT)[:, None] * stride_weight_out
        weight_mask = (input_mask[None, :]) & (tl.arange(0, BLOCK_SIZE_OUT)[:, None] < out_features)
        current_weight = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)

        logits += tl.sum(current_input[None, :] * current_weight, axis=1)

    bias_offsets = tl.arange(0, BLOCK_SIZE_OUT) * stride_bias_out
    bias_mask = tl.arange(0, BLOCK_SIZE_OUT) < out_features
    bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
    logits += bias

    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    log_sum_exp = tl.log(sum_exp)
    log_probs = logits - max_logit - log_sum_exp

    target_mask = tl.arange(0, BLOCK_SIZE_OUT) == target
    contribution = -tl.sum(log_probs * target_mask, axis=0)
    tl.atomic_add(loss_ptr, contribution / batch_size)

def fused_linear_cross_entropy(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    assert input.is_cuda and weight.is_cuda and bias.is_cuda and target.is_cuda
    batch_size, in_features = input.shape
    out_features, _ = weight.shape

    loss = torch.zeros(1, device=input.device, dtype=torch.float32)

    BLOCK_SIZE_IN = 128
    BLOCK_SIZE_OUT = triton.next_power_of_2(out_features)
    if BLOCK_SIZE_OUT > 4096:
        raise ValueError("Too many output features for this kernel implementation")

    grid = (batch_size,)
    fused_linear_xentropy_forward[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        target_ptr=target,
        loss_ptr=loss,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        stride_input_batch=input.stride(0),
        stride_input_feature=input.stride(1),
        stride_weight_out=weight.stride(0),
        stride_weight_in=weight.stride(1),
        stride_bias_out=bias.stride(0),
        BLOCK_SIZE_IN=BLOCK_SIZE_IN,
        BLOCK_SIZE_OUT=BLOCK_SIZE_OUT,
    )
    return loss