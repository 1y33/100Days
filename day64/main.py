import torch, time
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def geglu_kernel(input_ptr, output_ptr, numel: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    row = offsets // D
    col = offsets % D
    base_offset = row * (2 * D)
    x = tl.load(input_ptr + base_offset + col, mask=mask)
    gate = tl.load(input_ptr + base_offset + D + col, mask=mask)

    t = 0.7978845608 * (gate + 0.044715 * gate * gate * gate)
    exp_2t = tl.exp(2 * t)
    tanh_t = (exp_2t - 1.0) / (exp_2t + 1.0)
    gelu_gate = 0.5 * gate * (1.0 + tanh_t)
    out = x * gelu_gate

    tl.store(output_ptr + offsets, out, mask=mask)

def fused_geglu(input_tensor):
    N, twoD = input_tensor.shape
    D = twoD // 2
    output = torch.empty((N, D), device=input_tensor.device, dtype=input_tensor.dtype)
    numel = N * D
    BLOCK_SIZE = 256
    grid = lambda meta: ((numel + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    geglu_kernel[grid](input_tensor, output, numel, D, BLOCK_SIZE)
    return output

def torch_geglu(input_tensor):
    x, gate = input_tensor.chunk(2, dim=-1)
    return x * F.gelu(gate)

input_tensor = torch.randn(8192, 8192, device='cuda')

_ = fused_geglu(input_tensor)
_ = torch_geglu(input_tensor)

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = fused_geglu(input_tensor)
torch.cuda.synchronize()
fused_time = time.time() - start

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = torch_geglu(input_tensor)
torch.cuda.synchronize()
torch_time = time.time() - start

print("Fused Triton kernel time: {:.6f} sec".format(fused_time))
print("Torch baseline time: {:.6f} sec".format(torch_time))
