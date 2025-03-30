import torch
import triton
import triton.language as tl

@triton.jit
def relu_device_fn(x):
    return tl.maximum(0.0, x)

@triton.jit
def swish_device_fn(x):
    return x * tl.sigmoid(x)

@triton.jit
def gelu_device_fn(x):
    return 0.5 * x * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

def create_activation_kernel(device_fn):
    @triton.jit
    def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        output = device_fn(x)
        tl.store(output_ptr + offsets, output, mask=mask)
    return kernel

def create_activation_function(kernel, name):
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
            triton.Config({'BLOCK_SIZE': 1024}),
        ],
        key=['n_elements'],
    )
    def activation_fn(x):
        n_elements = x.numel()
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        kernel[grid](
            x.data_ptr(),
            output.data_ptr(),
            n_elements,
        )
        return output
    activation_fn.__name__ = name
    return activation_fn

relu_kernel = create_activation_kernel(relu_device_fn)
swish_kernel = create_activation_kernel(swish_device_fn)
gelu_kernel = create_activation_kernel(gelu_device_fn)

relu = create_activation_function(relu_kernel, "relu")
swish = create_activation_function(swish_kernel, "swish")
gelu = create_activation_function(gelu_kernel, "gelu")

def example():
    x = torch.randn(1024, 1024, device='cuda')
    y_relu = relu(x)
    y_swish = swish(x)
    y_gelu = gelu(x)
    print(f"Input shape: {x.shape}")
    print(f"ReLU output shape: {y_relu.shape}")
    print(f"Swish output shape: {y_swish.shape}")
    print(f"GELU output shape: {y_gelu.shape}")
    torch_relu = torch.nn.functional.relu(x)
    torch_gelu = torch.nn.functional.gelu(x)
    torch_swish = torch.nn.functional.silu(x)
    print(f"ReLU max error: {(y_relu - torch_relu).abs().max().item()}")
    print(f"Swish max error: {(y_swish - torch_swish).abs().max().item()}")
    print(f"GELU max error: {(y_gelu - torch_gelu).abs().max().item()}")

if __name__ == "__main__":
    example()
