import triton
import triton.language as tl

@triton.jit
def fused_layernorm_ff_dropout_kernel(
    x_ptr, out_ptr,
    gamma_ptr, beta_ptr,
    weight1_ptr, bias1_ptr,
    weight2_ptr, bias2_ptr,
    seed,
    dropout_p: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_offset = row_idx * N

    x = tl.load(x_ptr + row_offset + tl.arange(0, N))
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    norm = diff * tl.rsqrt(var + 1e-5)

    gamma = tl.load(gamma_ptr + tl.arange(0, N))
    beta = tl.load(beta_ptr + tl.arange(0, N))
    norm = norm * gamma + beta

    hidden = tl.zeros([M], dtype=x.dtype)
    for i in range(0, N, BLOCK):
        block_range = i + tl.arange(0, BLOCK)
        norm_block = norm[block_range]
        weight1_block = tl.load(
            weight1_ptr + i * M + tl.arange(0, BLOCK)[:, None] * M + tl.arange(0, M),
            mask=(i + tl.arange(0, BLOCK))[:, None] < N, other=0.0
        )
        hidden += tl.dot(norm_block, weight1_block)

    bias1 = tl.load(bias1_ptr + tl.arange(0, M))
    hidden += bias1

    SQRT_2_OVER_PI = 0.7978845608028654
    gelu_hidden = 0.5 * hidden * (1.0 + tl.tanh(SQRT_2_OVER_PI * (hidden + 0.044715 * hidden * hidden * hidden)))

    prng = tl.arange(0, M) + row_idx * M + seed
    rand_vals = ((1103515245 * prng + 12345) & 0x7fffffff) / 2147483647.0
    dropout_mask = rand_vals > dropout_p
    dropout_scale = 1.0 / (1.0 - dropout_p)
    dropped = gelu_hidden * dropout_mask * dropout_scale

    out = tl.zeros([N], dtype=x.dtype)
    for j in range(0, M, BLOCK):
        block_range = j + tl.arange(0, BLOCK)
        dropped_block = dropped[block_range]
        weight2_block = tl.load(
            weight2_ptr + j * N + tl.arange(0, BLOCK)[:, None] * N + tl.arange(0, N),
            mask=(j + tl.arange(0, BLOCK))[:, None] < M, other=0.0
        )
        out += tl.dot(dropped_block, weight2_block)

    bias2 = tl.load(bias2_ptr + tl.arange(0, N))
    out += bias2

    tl.store(out_ptr + row_offset + tl.arange(0, N), out)
