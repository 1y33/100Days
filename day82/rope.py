import torch
import triton
import triton.language as tl
import math

@triton.jit
def rope_kernel(q_ptr, cos_ptr, sin_ptr, stride_q0, stride_q1, stride_cos0, stride_cos1, seq_len: tl.constexpr, head_half: tl.constexpr, BLOCK_SEQ: tl.constexpr, BLOCK_HD: tl.constexpr):
    
    pid_seq = tl.program_id(0)
    pid_hd = tl.program_id(1)

    seq_offset = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    hd_offset = pid_hd * BLOCK_HD + tl.arange(0, BLOCK_HD)

    mask_seq = seq_offset < seq_len
    mask_hd = hd_offset < head_half

    q_ptrs = q_ptr + seq_offset[:, None] * stride_q0 + hd_offset[None, :] * (2 * stride_q1)

    q0 = tl.load(q_ptrs, mask=mask_seq[:, None] & mask_hd[None, :])
    q1 = tl.load(q_ptrs + stride_q1, mask=mask_seq[:, None] & mask_hd[None, :])

    cos_ptrs = cos_ptr + seq_offset[:, None] * stride_cos0 + hd_offset[None, :] * stride_cos1
    sin_ptrs = sin_ptr + seq_offset[:, None] * stride_cos0 + hd_offset[None, :] * stride_cos1

    cos_val = tl.load(cos_ptrs, mask=mask_seq[:, None] & mask_hd[None, :])
    sin_val = tl.load(sin_ptrs, mask=mask_seq[:, None] & mask_hd[None, :])

    out0 = q0 * cos_val - q1 * sin_val
    out1 = q0 * sin_val + q1 * cos_val

    tl.store(q_ptrs, out0, mask=mask_seq[:, None] & mask_hd[None, :])
    tl.store(q_ptrs + stride_q1, out1, mask=mask_seq[:, None] & mask_hd[None, :])

def apply_rope(q, cos, sin, BLOCK_SEQ=64, BLOCK_HD=32):
    
    seq_len, head_dim = q.shape
    assert head_dim % 2 == 0
    head_half = head_dim // 2

    grid = ((seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ, (head_half + BLOCK_HD - 1) // BLOCK_HD)

    q_contig = q.contiguous()

    rope_kernel[grid](q_contig, cos, sin, q_contig.stride(0), q_contig.stride(1), cos.stride(0), cos.stride(1), seq_len, head_half, BLOCK_SEQ, BLOCK_HD)
    return q_contig

if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'cuda'

    seq_len = 128
    head_dim = 64

    q = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)

    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    dim_idx = torch.arange(head_dim // 2, device=device, dtype=torch.float32).unsqueeze(0)
    inv_freq = 1.0 / (10000 ** (dim_idx / (head_dim // 2)))
    theta = positions * inv_freq

    cos = torch.cos(theta)
    sin = torch.sin(theta)

    q_transformed = apply_rope(q, cos, sin)
    print("Transformed q:")
    print(q_transformed)
