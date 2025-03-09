import triton
import triton.language as tl
import torch


@triton.jit
def layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W,b,r,
    mu,n_cols, eps,
    BLOCK_SIZE : tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y  += row_idx * Y_row_stride
    X  += row_idx * X_row_stride
    r  += row_idx
    mu += row_idx

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask = mask, other = 0).to(tl.float32)

    mean_X  = tl.sum(X_row,   axis = 0) / n_cols
    XX = tl.where(mask, X_row - mean_X, 0)
    row_var = tl.sum(XX * XX, axis = 0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store (r, inv_var)
    tl.store (mu, mean_X)
    output = (XX * inv_var) * W_row + b_row
    tl.store(Y + col_offsets, output, mask = mask)
pass


@triton.jit
def layernorm_backward(
    dY, dY_row_stride,
    X,   X_row_stride,
    W,b,r,
    mu,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx
    mu += row_idx

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask = mask, other = 0).to(tl.float32)
    b_row  = tl.load(b  + col_offsets, mask = mask, other = 0).to(tl.float32)

    inv_var = tl.load(r) .to(tl.float32)
    mean    = tl.load(mu).to(tl.float32)
    normed  = (X_row - mean) * inv_var
    dY_W = dY_row * W_row
    dX_row = dY_W - tl.sum(dY_W, axis = 0) / n_cols - normed * tl.sum(dY_W * normed, axis = 0) / n_cols
    dX_row = dX_row * inv_var
    tl.store(dY + col_offsets, dX_row, mask = mask)
pass


class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        device = X.device
        Y  = torch.empty((n_rows, n_cols), dtype = X.dtype, device = device)
        r  = torch.empty(n_rows, dtype = torch.float32, device = device)
        mu = torch.empty(n_rows, dtype = torch.float32, device = device)

        with torch_cuda_device(device):
            layernorm_forward[(n_rows,)](
                Y, Y.stride(0),
                X, X.stride(0),
                W,
                b,
                r,
                mu,
                n_cols, eps,
                BLOCK_SIZE = BLOCK_SIZE,
                num_warps  = num_warps,
            )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.save_for_backward(X, W, b, r, mu)
        return Y.view(*shape)
    pass

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, b, r, mu = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        with torch_cuda_device(dY.device):
            layernorm_backward[(n_rows,)](
                dY, dY.stride(0),
                X,  X .stride(0),
                W,
                b,
                r,
                mu,
                n_cols, ctx.eps,
                BLOCK_SIZE = ctx.BLOCK_SIZE,
                num_warps  = ctx.num_warps,
            )
        dX = dY.view(*shape)
        return dX, None, None, None, None
    pass
pass
