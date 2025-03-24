import torch
import triton
import triton.language as tl

@triton.jit
def adam_fp8_kernel(
    param_ptr, grad_ptr, m_ptr, v_ptr, lr_ptr, 
    beta1, beta2, eps, step,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < tl.numel(param_ptr)
    
    param = tl.load(param_ptr + offset, mask=mask, other=0.0).to(tl.float16)
    grad = tl.load(grad_ptr + offset, mask=mask, other=0.0).to(tl.float16)
    m = tl.load(m_ptr + offset, mask=mask, other=0.0).to(tl.float16)
    v = tl.load(v_ptr + offset, mask=mask, other=0.0).to(tl.float16)
    lr = tl.load(lr_ptr + offset, mask=mask, other=0.0).to(tl.float16)
    
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m_new / (1 - beta1 ** step)
    v_hat = v_new / (1 - beta2 ** step)
    update = m_hat / (tl.sqrt(v_hat) + eps)
    param_new = param - lr * update
    
    param_new_fp8 = param_new.to(tl.float8_e4m3)
    m_new_fp8 = m_new.to(tl.float8_e4m3)
    v_new_fp8 = v_new.to(tl.float8_e4m3)
    
    tl.store(param_ptr + offset, param_new_fp8, mask=mask)
    tl.store(m_ptr + offset, m_new_fp8, mask=mask)
    tl.store(v_ptr + offset, v_new_fp8, mask=mask)

def adam_fp8(param, grad, m, v, lr, beta1=0.9, beta2=0.999, eps=1e-8, step=1):
    BLOCK_SIZE = 1024
    n = param.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    adam_fp8_kernel[grid](
        param, grad, m, v, lr,
        beta1, beta2, eps, step,
        BLOCK_SIZE=BLOCK_SIZE
    )
