import torch
import triton
import triton.language as tl

@triton.jit
def reduce_kernel(K, V, A_ptr, b_ptr, N: tl.constexpr, D: tl.constexpr):
    acc_A = tl.zeros([D, D], dtype=tl.float32)
    acc_b = tl.zeros([D], dtype=tl.float32)
    for j in range(N):
        k = tl.load(K + j * D)
        v = tl.load(V + j * D)
        k_phi = tl.relu(k) + 1.0
        for i in range(D):
            for jj in range(D):
                acc_A[i, jj] += k_phi[i] * v[jj]
        acc_b += k_phi
    tl.store(A_ptr, acc_A)
    tl.store(b_ptr, acc_b)

@triton.jit
def attention_kernel(Q, A_ptr, b_ptr, Out, N: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    q = tl.load(Q + pid * D)
    q_phi = tl.relu(q) + 1.0
    out_vec = tl.zeros([D], dtype=tl.float32)
    for i in range(D):
        a_row = tl.load(A_ptr + i * D)
        out_vec[i] = tl.dot(q_phi, a_row)
    denom = tl.dot(q_phi, tl.load(b_ptr))
    tl.store(Out + pid * D, out_vec / denom)

def linear_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Input tensors must be on CUDA"
    N, D = Q.shape
    A = torch.empty((D, D), device='cuda', dtype=torch.float32)
    b = torch.empty((D,), device='cuda', dtype=torch.float32)
    reduce_kernel[(1,)](K, V, A, b, N, D)
    Out = torch.empty_like(Q)
    attention_kernel[(N,)](Q, A, b, Out, N, D)
    return Out

if __name__ == "__main__":
    N = 1024
    D = 64
    Q = torch.randn((N, D), device='cuda', dtype=torch.float32)
    K = torch.randn((N, D), device='cuda', dtype=torch.float32)
    V = torch.randn((N, D), device='cuda', dtype=torch.float32)
    Out = linear_attention(Q, K, V)
    print("Output shape:", Out.shape)
