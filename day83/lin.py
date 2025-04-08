import triton
import triton.language as tl

@triton.jit
def compute_S_kernel(K, V, S, B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr, d_v: tl.constexpr, stride_B_K: tl.constexpr, stride_H_K: tl.constexpr, stride_N_K: tl.constexpr, stride_D_K: tl.constexpr, stride_B_V: tl.constexpr, stride_H_V: tl.constexpr, stride_N_V: tl.constexpr, stride_dv_V: tl.constexpr, stride_B_S: tl.constexpr, stride_H_S: tl.constexpr, stride_D_S: tl.constexpr, stride_dv_S: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    s_ptr = S + pid_b * stride_B_S + pid_h * stride_H_S + pid_d * stride_D_S
    acc = tl.zeros((d_v,), dtype=tl.float32)
    for j in range(0, N, BLOCK_N):
        j_idx = j + tl.arange(0, BLOCK_N)
        mask = j_idx < N
        k_ptr = K + pid_b * stride_B_K + pid_h * stride_H_K + j_idx * stride_N_K + pid_d
        k_vals = tl.load(k_ptr, mask=mask, other=0.0)
        phi_k = tl.where(k_vals > 0.0, k_vals + 1.0, tl.exp(k_vals))
        v_ptr = V + pid_b * stride_B_V + pid_h * stride_H_V + j_idx[:, None] * stride_N_V
        n_ids = tl.arange(0, d_v)
        mask_v = mask[:, None]
        v_vals = tl.load(v_ptr + n_ids, mask=mask_v, other=0.0)
        prod = phi_k[:, None] * v_vals
        acc += tl.sum(prod, axis=0)
    tl.store(s_ptr, acc)

@triton.jit
def compute_z_kernel(K, z, B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr, stride_B_K: tl.constexpr, stride_H_K: tl.constexpr, stride_N_K: tl.constexpr, stride_D_K: tl.constexpr, stride_z: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    z_ptr = z + pid_b * (H * D) + pid_h * D + pid_d
    acc = 0.0
    for j in range(0, N, BLOCK_N):
        j_idx = j + tl.arange(0, BLOCK_N)
        mask = j_idx < N
        k_ptr = K + pid_b * stride_B_K + pid_h * stride_H_K + j_idx * stride_N_K + pid_d
        k_vals = tl.load(k_ptr, mask=mask, other=0.0)
        phi_k = tl.where(k_vals > 0.0, k_vals + 1.0, tl.exp(k_vals))
        acc += tl.sum(phi_k)
    tl.store(z_ptr, acc)

@triton.jit
def compute_attention_kernel(Q, S, z, Y, B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr, d_v: tl.constexpr, eps: tl.constexpr, stride_B_Q: tl.constexpr, stride_H_Q: tl.constexpr, stride_N_Q: tl.constexpr, stride_D_Q: tl.constexpr, stride_B_S: tl.constexpr, stride_H_S: tl.constexpr, stride_D_S: tl.constexpr, stride_B_z: tl.constexpr, stride_H_z: tl.constexpr, stride_B_Y: tl.constexpr, stride_H_Y: tl.constexpr, stride_N_Y: tl.constexpr, stride_dv_Y: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    q_ptr = Q + pid_b * stride_B_Q + pid_h * stride_H_Q + pid_n * stride_N_Q
    q_vec = tl.load(q_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D)
    phi_q = tl.where(q_vec > 0.0, q_vec + 1.0, tl.exp(q_vec))
    num = tl.zeros((d_v,), dtype=tl.float32)
    den = 0.0
    for d in range(D):
        s_ptr = S + pid_b * stride_B_S + pid_h * stride_H_S + d * stride_D_S
        s_val = tl.load(s_ptr + tl.arange(0, d_v), mask=tl.arange(0, d_v) < d_v)
        num += phi_q[d] * s_val
        z_ptr = z + pid_b * (H * D) + pid_h * D + d
        z_val = tl.load(z_ptr)
        den += phi_q[d] * z_val
    out = num / (den + eps)
    y_ptr = Y + pid_b * stride_B_Y + pid_h * stride_H_Y + pid_n * stride_N_Y
    tl.store(y_ptr + tl.arange(0, d_v), out, mask=tl.arange(0, d_v) < d_v)

import torch

B, H, N, D, d_v = 2, 4, 128, 64, 32
eps = 1e-6
K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
V = torch.randn(B, H, N, d_v, device='cuda', dtype=torch.float32)
Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
S = torch.empty(B, H, D, d_v, device='cuda', dtype=torch.float32)
z = torch.empty(B, H, D, device='cuda', dtype=torch.float32)
Y = torch.empty(B, H, N, d_v, device='cuda', dtype=torch.float32
                )
stride_B_K, stride_H_K, stride_N_K, stride_D_K = K.stride()
stride_B_V, stride_H_V, stride_N_V, stride_dv_V = V.stride()
stride_B_Q, stride_H_Q, stride_N_Q, stride_D_Q = Q.stride()
stride_B_S, stride_H_S, stride_D_S, stride_dv_S = S.stride()
stride_B_Y, stride_H_Y, stride_N_Y, stride_dv_Y = Y.stride()
stride_z = z.stride()[0]

BLOCK_N = 32
grid_S = (B, H, D)

compute_S_kernel[grid_S](K, V, S, B, H, N, D, 
                         d_v, stride_B_K, stride_H_K, stride_N_K, stride_D_K,
                         stride_B_V, stride_H_V, stride_N_V, stride_dv_V,
                         stride_B_S, stride_H_S, stride_D_S, stride_dv_S, 
                         BLOCK_N)
grid_z = (B, H, D)

compute_z_kernel[grid_z](K, z, B, H, N, D, 
                         stride_B_K, stride_H_K, stride_N_K, stride_D_K, stride_z,
                         BLOCK_N)
grid_attn = (B, H, N)

compute_attention_kernel[grid_attn](Q, S, z, Y, B, H, N, D, 
                                    d_v, eps,
                                    stride_B_Q, stride_H_Q, stride_N_Q, stride_D_Q,
                                    stride_B_S, stride_H_S, stride_D_S, stride_B_z, s
                                    tride_H_z, stride_B_Y, stride_H_Y, stride_N_Y, stride_dv_Y)
print("Output shape:", Y.shape)
