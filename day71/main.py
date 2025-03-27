import torch
import time
import numpy as np
import matplotlib.pyplot as plt

import triton
import triton.language as tl

def complex_loss_pytorch(pred, target, alpha=0.5, gamma=2.0, lambda_reg=0.01):
    l1_loss = torch.abs(pred - target).mean(dim=1)
    l2_loss = torch.square(pred - target).mean(dim=1)
    combined_loss = alpha * l1_loss + (1 - alpha) * l2_loss
    error_scale = torch.clamp(combined_loss.detach(), 0.1, 10.0)
    focal_weight = error_scale.pow(gamma)
    weighted_loss = (combined_loss * focal_weight).mean()
    reg_term = lambda_reg * torch.square(pred).mean()
    return weighted_loss + reg_term

@triton.jit
def complex_loss_kernel(
    pred_ptr, target_ptr, l1_ptr, l2_ptr, reg_ptr,
    batch_size, feature_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= batch_size:
        return
    row_start = pid * feature_dim
    l1_loss_sum = 0.0
    l2_loss_sum = 0.0
    reg_term_sum = 0.0
    count = 0
    for start_idx in range(0, feature_dim, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < feature_dim
        pred_offsets = row_start + offsets
        target_offsets = row_start + offsets
        pred = tl.load(pred_ptr + pred_offsets, mask=mask, other=0.0)
        target = tl.load(target_ptr + target_offsets, mask=mask, other=0.0)
        l1_loss = tl.abs(pred - target)
        l1_loss_sum += tl.sum(l1_loss * mask)
        l2_loss = (pred - target) ** 2
        l2_loss_sum += tl.sum(l2_loss * mask)
        reg_term = pred ** 2
        reg_term_sum += tl.sum(reg_term * mask)
        count += tl.sum(mask)
    l1_loss_mean = l1_loss_sum / count
    l2_loss_mean = l2_loss_sum / count
    reg_term_mean = reg_term_sum / count
    tl.store(l1_ptr + pid, l1_loss_mean)
    tl.store(l2_ptr + pid, l2_loss_mean)
    tl.store(reg_ptr + pid, reg_term_mean)

def complex_loss_triton(pred, target, alpha=0.5, gamma=2.0, lambda_reg=0.01):
    pred = pred.contiguous()
    target = target.contiguous()
    batch_size, feature_dim = pred.shape
    l1_loss = torch.empty(batch_size, device=pred.device, dtype=pred.dtype)
    l2_loss = torch.empty(batch_size, device=pred.device, dtype=pred.dtype)
    reg_term = torch.empty(batch_size, device=pred.device, dtype=pred.dtype)
    BLOCK_SIZE = min(1024, feature_dim)
    complex_loss_kernel[(batch_size,)](
        pred.data_ptr(), target.data_ptr(), 
        l1_loss.data_ptr(), l2_loss.data_ptr(), reg_term.data_ptr(),
        batch_size, feature_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    combined_loss = alpha * l1_loss + (1 - alpha) * l2_loss
    error_scale = torch.clamp(combined_loss, 0.1, 10.0)
    focal_weight = error_scale.pow(gamma)
    weighted_loss = (combined_loss * focal_weight).mean()
    reg_term = lambda_reg * reg_term.mean()
    return weighted_loss + reg_term

def benchmark(func, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    timings = []
    iterations = 100
    for _ in range(iterations):
        start.record()
        func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return sum(timings) / len(timings)

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a GPU.")
        return
    torch.manual_seed(42)
    batch_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    feature_dim = 256
    pytorch_times = []
    triton_times = []
    speedups = []
    print("Benchmarking Complex Loss Function: PyTorch vs Triton")
    print("-" * 60)
    print(f"{'Batch Size':>10} | {'PyTorch (ms)':>12} | {'Triton (ms)':>11} | {'Speedup':>8} | {'Error':>8}")
    print("-" * 60)
    for batch_size in batch_sizes:
        pred = torch.randn(batch_size, feature_dim, device='cuda', requires_grad=True)
        target = torch.randn(batch_size, feature_dim, device='cuda')
        pred_triton = pred.detach().clone().requires_grad_(True)
        pytorch_loss = complex_loss_pytorch(pred, target)
        triton_loss = complex_loss_triton(pred_triton, target)
        error = abs(pytorch_loss.item() - triton_loss.item())
        rel_error = error / abs(pytorch_loss.item()) if pytorch_loss.item() != 0 else error
        pytorch_time = benchmark(complex_loss_pytorch, pred, target)
        pytorch_times.append(pytorch_time)
        triton_time = benchmark(complex_loss_triton, pred_triton, target)
        triton_times.append(triton_time)
        speedup = pytorch_time / triton_time
        speedups.append(speedup)
        print(f"{batch_size:>10} | {pytorch_time:>12.3f} | {triton_time:>11.3f} | {speedup:>8.2f}x | {rel_error:>8.2e}")
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(batch_sizes, pytorch_times, marker='o', label='PyTorch')
    plt.plot(batch_sizes, triton_times, marker='x', label='Triton')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (ms)')
    plt.title('Performance Comparison: PyTorch vs Triton')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(batch_sizes, speedups, marker='s', color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup (x times)')
    plt.title('Triton Speedup over PyTorch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pytorch_vs_triton_performance.png')
    plt.show()
    print("\nResults summary:")
    print(f"Maximum speedup: {max(speedups):.2f}x at batch size {batch_sizes[speedups.index(max(speedups))]}")
    print(f"Minimum speedup: {min(speedups):.2f}x at batch size {batch_sizes[speedups.index(min(speedups))]}")
    print(f"Average speedup: {sum(speedups)/len(speedups):.2f}x")

if __name__ == "__main__":
    main()
