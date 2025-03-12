import torch
import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, mask_ptr, out_ptr,
    # Matrix dimensions
    batch_size, num_heads, seq_len, head_dim, block_size,
    # Block sparse mask info
    num_blocks, blocks_ptr,  
    # Strides
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    scale,
):
    pid_block = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    block_start_idx = tl.load(blocks_ptr + pid_block * 2)
    block_end_idx = tl.load(blocks_ptr + pid_block * 2 + 1)
    
    query_start = (block_start_idx // (seq_len // block_size)) * block_size
    query_end = query_start + block_size
    key_start = (block_start_idx % (seq_len // block_size)) * block_size
    key_end = key_start + block_size
    
    offs_q = pid_batch * stride_qb + pid_head * stride_qh
    offs_k = pid_batch * stride_kb + pid_head * stride_kh
    offs_v = pid_batch * stride_vb + pid_head * stride_vh
    offs_o = pid_batch * stride_ob + pid_head * stride_oh
    
    q_block_ptr = q_ptr + offs_q + query_start * stride_qs
    k_block_ptr = k_ptr + offs_k + key_start * stride_ks
    v_block_ptr = v_ptr + offs_v + key_start * stride_vs
    
    q_block = tl.load(q_block_ptr + tl.arange(0, block_size)[:, None] * stride_qs + 
                     tl.arange(0, head_dim)[None, :] * stride_qd)
    k_block = tl.load(k_block_ptr + tl.arange(0, block_size)[:, None] * stride_ks + 
                     tl.arange(0, head_dim)[None, :] * stride_kd)
    v_block = tl.load(v_block_ptr + tl.arange(0, block_size)[:, None] * stride_vs + 
                     tl.arange(0, head_dim)[None, :] * stride_vd)
    
    scores = tl.dot(q_block, tl.trans(k_block))
    scores = scores * scale
    
    causal_mask = tl.arange(0, block_size)[:, None] >= tl.arange(0, block_size)[None, :]
    scores = scores * causal_mask + (1 - causal_mask) * -1e9
    
    scores_max = tl.max(scores, axis=1, keepdims=True)
    scores = scores - scores_max
    scores_exp = tl.exp(scores)
    scores_sum = tl.sum(scores_exp, axis=1, keepdims=True)
    attention = scores_exp / scores_sum
    
    output = tl.dot(attention, v_block)
    
    out_ptr = out_ptr + offs_o + query_start * stride_os
    tl.store(out_ptr + tl.arange(0, block_size)[:, None] * stride_os + 
             tl.arange(0, head_dim)[None, :] * stride_od, output)


def block_sparse_attention(q, k, v, block_indices, block_size=16, causal=True):
    batch_size, num_heads, seq_len, head_dim = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    output = torch.zeros_like(q)
    
    blocks = torch.tensor(block_indices, device=q.device, dtype=torch.int32)
    num_blocks = blocks.size(0) // 2
    
    scale = head_dim ** -0.5
    
    mask = torch.zeros((1,), device=q.device)
    
    grid = (num_blocks, batch_size, num_heads)
    
    block_sparse_attention_kernel[grid](
        q, k, v, mask, output,
        batch_size, num_heads, seq_len, head_dim, block_size,
        num_blocks, blocks,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale
    )
    
    return output


import torch

batch_size, num_heads, seq_len, head_dim = 2, 4, 64, 32
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)

block_indices = [0, 0, 0, 1, 1, 1, 2, 2]

output = block_sparse_attention(q, k, v, block_indices, block_size=16)
print(output.shape)