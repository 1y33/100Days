import torch
import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    param_ptr,
    grad_ptr,
    momentum_ptr,
    lr,
    weight_decay,
    momentum_factor,
    dampening,
    nesterov,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    params = tl.load(param_ptr + offsets, mask=mask)
    grads = tl.load(grad_ptr + offsets, mask=mask)
    if weight_decay != 0.0:
        grads = grads + weight_decay * params
    if momentum_factor != 0.0:
        momentum_buf = tl.load(momentum_ptr + offsets, mask=mask)
        momentum_buf = momentum_factor * momentum_buf + (1.0 - dampening) * grads
        tl.store(momentum_ptr + offsets, momentum_buf, mask=mask)
        if nesterov:
            grads = grads + momentum_factor * momentum_buf
        else:
            grads = momentum_buf
    params = params - lr * grads
    tl.store(param_ptr + offsets, params, mask=mask)

def sgd_update(
    params,
    grads,
    momentum_buffer=None,
    lr=0.01,
    weight_decay=0.0,
    momentum=0.0,
    dampening=0.0,
    nesterov=False,
):
    n_elements = params.numel()
    if momentum != 0.0 and momentum_buffer is None:
        momentum_buffer = torch.zeros_like(params)
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    sgd_kernel[grid, 1](
        params.data_ptr(),
        grads.data_ptr(),
        momentum_buffer.data_ptr() if momentum != 0.0 else 0,
        lr,
        weight_decay,
        momentum,
        dampening,
        1 if nesterov else 0,
        n_elements,
        BLOCK_SIZE,
    )
    return params, momentum_buffer

def example():
    params = torch.randn(10000, device='cuda')
    grads = torch.randn(10000, device='cuda')
    momentum_buffer = torch.zeros_like(params)
    updated_params, updated_momentum = sgd_update(
        params, 
        grads,
        momentum_buffer,
        lr=0.01,
        weight_decay=0.0001,
        momentum=0.9,
        nesterov=True
    )
    print(f"Updated {params.shape} parameters using Triton SGD kernel")
    
if __name__ == "__main__":
    example()
