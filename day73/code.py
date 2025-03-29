import triton
import triton.language as tl
import numpy as np

@triton.jit
def ddim_step_kernel(
    x_ptr,
    eps_ptr,
    out_ptr,
    alpha_t: tl.constexpr,
    alpha_t_prev: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1024
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    eps = tl.load(eps_ptr + offsets, mask=mask)
    
    sqrt_alpha_t = tl.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = tl.sqrt(1 - alpha_t)
    sqrt_alpha_t_prev = tl.sqrt(alpha_t_prev)
    sqrt_one_minus_alpha_t_prev = tl.sqrt(1 - alpha_t_prev)

    x0 = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
    new_x = sqrt_alpha_t_prev * x0 + sqrt_one_minus_alpha_t_prev * eps

    tl.store(out_ptr + offsets, new_x, mask=mask)

def ddim_sampling_step(x: np.ndarray, eps: np.ndarray, alpha_t: float, alpha_t_prev: float):
    x = np.ascontiguousarray(x.astype(np.float32))
    eps = np.ascontiguousarray(eps.astype(np.float32))
    out = np.empty_like(x)

    n_elements = x.size
    grid = (triton.cdiv(n_elements, 1024),)

    ddim_step_kernel[grid](
        x_ptr=x,
        eps_ptr=eps,
        out_ptr=out,
        alpha_t=alpha_t,
        alpha_t_prev=alpha_t_prev,
        n_elements=n_elements,
        BLOCK_SIZE=1024
    )
    return out

if __name__ == '__main__':
    N = 4096
    x = np.random.randn(N).astype(np.float32)
    eps = np.random.randn(N).astype(np.float32)
    alpha_t = 0.9
    alpha_t_prev = 0.85

    x_prev = ddim_sampling_step(x, eps, alpha_t, alpha_t_prev)
    print("Updated sample:", x_prev)
