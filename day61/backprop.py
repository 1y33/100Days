import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, 
    batch_size, num_classes,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
        
    input_row_ptr = input_ptr + pid * num_classes
    logits = tl.load(input_row_ptr + tl.arange(0, num_classes))
    
    logits_max = tl.max(logits, axis=0)
    logits = logits - logits_max
    
    exp_logits = tl.exp(logits)
    
    sum_exp = tl.sum(exp_logits, axis=0)
    
    probs = exp_logits / sum_exp
    
    output_row_ptr = output_ptr + pid * num_classes
    tl.store(output_row_ptr + tl.arange(0, num_classes), probs)

@triton.jit
def cross_entropy_forward_kernel(
    loss_ptr, logits_ptr, targets_ptr,
    batch_size, num_classes,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    logits_row_ptr = logits_ptr + pid * num_classes
    logits = tl.load(logits_row_ptr + tl.arange(0, num_classes))
    
    target_idx = tl.load(targets_ptr + pid)
    
    logits_max = tl.max(logits, axis=0)
    logits = logits - logits_max
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    log_sum_exp = tl.log(sum_exp)
    log_probs = logits - log_sum_exp
    
    target_mask = tl.arange(0, num_classes) == target_idx
    target_log_prob = tl.sum(log_probs * target_mask)
    
    loss = -target_log_prob
    
    tl.store(loss_ptr + pid, loss)

@triton.jit
def cross_entropy_backward_kernel(
    grad_logits_ptr, logits_ptr, targets_ptr, grad_output_ptr,
    batch_size, num_classes,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    logits_row_ptr = logits_ptr + pid * num_classes
    logits = tl.load(logits_row_ptr + tl.arange(0, num_classes))
    
    target_idx = tl.load(targets_ptr + pid)
    
    grad_output = tl.load(grad_output_ptr + pid)
    
    logits_max = tl.max(logits, axis=0)
    logits = logits - logits_max
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp
    
    target_one_hot = tl.where(tl.arange(0, num_classes) == target_idx, 1.0, 0.0)
    
    grad_logits = (probs - target_one_hot) * grad_output
    
    grad_row_ptr = grad_logits_ptr + pid * num_classes
    tl.store(grad_row_ptr + tl.arange(0, num_classes), grad_logits)

def cross_entropy_forward(logits, targets):
    batch_size, num_classes = logits.shape
    loss = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
    
    grid = (batch_size,)
    block_size = min(num_classes, 1024)
    
    cross_entropy_forward_kernel[grid](
        loss, logits, targets,
        batch_size, num_classes,
        BLOCK_SIZE=block_size
    )
    
    return loss.mean()

def cross_entropy_backward(grad_output, logits, targets):
    batch_size, num_classes = logits.shape
    grad_logits = torch.empty_like(logits)
    
    grad_output_expanded = torch.ones(batch_size, device=logits.device, dtype=logits.dtype) * (grad_output / batch_size)
    
    grid = (batch_size,)
    block_size = min(num_classes, 1024)
    
    cross_entropy_backward_kernel[grid](
        grad_logits, logits, targets, grad_output_expanded,
        batch_size, num_classes,
        BLOCK_SIZE=block_size
    )
    
    return grad_logits

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        ctx.save_for_backward(logits, targets)
        loss = cross_entropy_forward(logits, targets)
        return loss
        
    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        grad_logits = cross_entropy_backward(grad_output, logits, targets)
        return grad_logits, None

def cross_entropy_loss(logits, targets):
    return CrossEntropyLoss.apply(logits, targets)