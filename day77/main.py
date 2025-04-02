import torch
import triton
import triton.language as tl
import time

# A simplified RetNet kernel: a decaying cumulative sum.
# Given an input sequence x and a decay factor alpha,
# it computes: y[0] = x[0] and for i>0, y[i] = x[i] + alpha * y[i-1]
# Note: This kernel assumes the sequence length (N) is known at compile-time.
@triton.jit
def retnet_kernel(x_ptr, y_ptr, N: tl.constexpr, alpha: tl.constexpr):
    # we use a single program (grid = (1,)) to process the full sequence sequentially.
    acc = tl.zeros([1], dtype=tl.float32)
    # Process each element in sequence.
    for i in range(N):
        # Load the i-th element from input.
        x_val = tl.load(x_ptr + i)
        # Compute the recurrent relation.
        acc = x_val + alpha * acc
        # Store the result.
        tl.store(y_ptr + i, acc)

# A CPU reference implementation for testing correctness and timing.
def retnet_cpu(x, alpha):
    y = torch.empty_like(x)
    acc = 0.0
    for i in range(x.shape[0]):
        acc = x[i].item() + alpha * acc
        y[i] = acc
    return y

def main():
    # Parameters
    N = 1024  # Sequence length (must match the kernel compile-time constant)
    alpha = 0.9
    # Create a random input tensor on the GPU.
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)

    # Define a grid that launches one program instance (since the kernel is sequential).
    grid = lambda meta: (1,)

    # Warm-up: launch the kernel once to compile and warm up.
    retnet_kernel[grid](x, y, N, alpha)
    torch.cuda.synchronize()

    # Time the Triton kernel using CUDA events.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    retnet_kernel[grid](x, y, N, alpha)
    end_event.record()
    torch.cuda.synchronize()
    triton_time = start_event.elapsed_time(end_event)  # milliseconds

    # Run the CPU version for comparison.
    x_cpu = x.cpu()
    start = time.time()
    y_cpu = retnet_cpu(x_cpu, alpha)
    cpu_time = (time.time() - start) * 1000  # convert to ms

    # Verify correctness.
    y_ref = y_cpu.to(device='cuda')
    if torch.allclose(y, y_ref, atol=1e-5):
        print("Results match.")
    else:
        print("Results differ!")

    print("Triton kernel time (ms):", triton_time)
    print("CPU cumulative sum time (ms):", cpu_time)

if __name__ == '__main__':
    main()
