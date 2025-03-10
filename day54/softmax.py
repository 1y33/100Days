import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_start + col_offsets
    row_mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=row_mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=row_mask)

def triton_softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    y = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda
    
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    softmax_kernel[(n_rows,)](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return y

x = torch.randn(10000, 1000, device='cuda')
triton_result = triton_softmax(x)
torch_result = torch.softmax(x, axis=1)

print(f"Max error: {torch.max(torch.abs(triton_result - torch_result)):.2e}")