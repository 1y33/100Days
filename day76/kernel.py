import torch
import triton
import triton.language as tl

@triton.jit
def moe_kernel(
    input_ptr,
    gate_weight_ptr,
    experts_ptr,
    output_ptr,
    num_tokens,
    hidden_size,
    num_experts,
    top_k,
    input_token_stride,
    input_hidden_stride,
    expert_stride,
    expert_hidden_stride,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return
    
    input_offset = token_idx * input_token_stride
    input = tl.load(input_ptr + input_offset + tl.arange(0, BLOCK_SIZE) * input_hidden_stride,
                   mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=0.0)
    
    gate_logits = tl.zeros((num_experts,), dtype=tl.float32)
    for expert in range(num_experts):
        gate_w = tl.load(gate_weight_ptr + expert * hidden_size + tl.arange(0, BLOCK_SIZE),
                         mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=0.0)
        logit = tl.sum(input * gate_w)
        gate_logits = tl.store(gate_logits + expert, logit)
    
    max_logit = tl.max(gate_logits)
    exp_logits = tl.exp(gate_logits - max_logit)
    sum_exp = tl.sum(exp_logits)
    probs = exp_logits / sum_exp
    
    topk_values, topk_indices = tl.topk(probs, top_k)
    
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(top_k):
        expert_idx = topk_indices[i]
        weight = topk_values[i]
        
        expert_offset = expert_idx * expert_stride
        expert_output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(hidden_size):
            w = tl.load(experts_ptr + expert_offset + j * expert_hidden_stride + tl.arange(0, BLOCK_SIZE),
                        mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=0.0)
            expert_output += input[j] * w
        
        output += weight * expert_output
    
    tl.store(output_ptr + token_idx * input_token_stride + tl.arange(0, BLOCK_SIZE) * input_hidden_stride,
            output, mask=tl.arange(0, BLOCK_SIZE) < hidden_size)

def moe_layer(input: torch.Tensor, gate: torch.Tensor, experts: torch.Tensor, top_k: int):
    assert experts.shape[0] >= top_k, "Number of experts must be >= top_k"
    output = torch.empty_like(input)
    hidden_size = input.size(1)
    num_tokens = input.size(0)
    num_experts = gate.size(1)
    
    # Ensure block size is a power of two for optimal performance
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096
    
    moe_kernel[(num_tokens,)](
        input_ptr=input,
        gate_weight_ptr=gate,
        experts_ptr=experts,
        output_ptr=output,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        input_token_stride=input.stride(0),
        input_hidden_stride=input.stride(1),
        expert_stride=experts.stride(0),
        expert_hidden_stride=experts.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output