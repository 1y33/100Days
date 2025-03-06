import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    sqrt_2_over_pi = tl.sqrt(2.0 / tl.math.pi)
    cdf = 0.5 * (1.0 + tl.tanh(sqrt_2_over_pi * (x + 0.044715 * (x ** 3))))
    output = x * cdf
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def fused_add_multiply_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    output = (a + b) * c
    tl.store(output_ptr + offsets, output, mask=mask)

class GELUTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            with torch.cuda.amp.autocast():
                output = GELUTriton.apply(x)
            grad_input = torch.autograd.grad(
                output, x, grad_output, create_graph=True
            )[0]
        return grad_input

def fused_add_multiply(a, b, c):
    assert a.shape == b.shape == c.shape
    a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
    output = torch.empty_like(a)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_add_multiply_kernel[grid](a, b, c, output, n_elements, BLOCK_SIZE=1024)
    return output

class TritonNN(torch.nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = GELUTriton.apply(x)
        
        residual = x
        a = x
        b = torch.ones_like(x) * 0.5  
        c = torch.ones_like(x) * 1.5
        x = fused_add_multiply(a, b, c)
        x += residual  
        
        return self.fc2(x)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TritonNN(784, 256, 10).to(device)
    x = torch.randn(32, 784).to(device)
    output = model(x)
    print("Output shape:", output.shape)
    print("Output values:", output[0, :5])