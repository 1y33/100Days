import torch
import triton
import triton.language as tl
import time
import math

@triton.jit
def _ddpm_kernel(x_ptr, eps_ptr, out_ptr,
                 alpha: tl.constexpr, beta: tl.constexpr, alpha_bar: tl.constexpr,
                 n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    eps = tl.load(eps_ptr + offsets, mask=mask)
    
    inv_sqrt_alpha = 1.0 / tl.sqrt(alpha)
    scale_eps = beta / tl.sqrt(1.0 - alpha_bar)
    out = inv_sqrt_alpha * (x - scale_eps * eps)
    
    tl.store(out_ptr + offsets, out, mask=mask)

def ddpm_kernel_update(x: torch.Tensor, epsilon_pred: torch.Tensor, alpha: float, beta: float, alpha_bar: float):
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _ddpm_kernel[grid](x, epsilon_pred, out, alpha, beta, alpha_bar, n_elements, BLOCK_SIZE=1024)
    return out

def normal_update(x: torch.Tensor, epsilon_pred: torch.Tensor, alpha: float, beta: float, alpha_bar: float):
    inv_sqrt_alpha = 1 / torch.sqrt(torch.tensor(alpha, device=x.device))
    scale_eps = beta / torch.sqrt(torch.tensor(1 - alpha_bar, device=x.device))
    return inv_sqrt_alpha * (x - scale_eps * epsilon_pred)

######################
import torch.nn as nn
class SimpleUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleUNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.time_emb = nn.Embedding(1000, hidden_dim)

    def forward(self, x, t):
        temb = self.time_emb(t)
        h = self.fc1(x) + temb
        h = torch.relu(h)
        h = self.fc2(h)
        return h
########
def sample_ddpm(model, func, scheduler, shape, device='cpu'):
    x = torch.randn(shape, device=device)
    T = scheduler['T']
    betas = scheduler['betas']
    alphas = scheduler['alphas']
    alpha_bars = scheduler['alpha_bars']

    for t in reversed(range(T)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        epsilon_pred = model(x, t_tensor)
        # Convert tensors to Python scalars before passing to the update function
        beta = betas[t].item()
        alpha = alphas[t].item()
        alpha_bar = alpha_bars[t].item()
        
        x = func(x, epsilon_pred, alpha, beta, alpha_bar)
        
        if t > 0:
            noise = torch.randn_like(x)
            sigma = math.sqrt(beta)
            x = x + sigma * noise
    return x


########3
def get_scheduler(T, beta_start=0.0001, beta_end=0.02, device='cpu'):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        'T': T,
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars,
    }



def benchmark_update(func, x, eps, alpha, beta, alpha_bar, iterations=100):
    for _ in range(10):
        _ = func(x, eps, alpha, beta, alpha_bar)
    
    if x.device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iterations):
            _ = func(x, eps, alpha, beta, alpha_bar)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        avg_time = elapsed_ms / iterations
    else:
        start_time = time.time()
        for _ in range(iterations):
            _ = func(x, eps, alpha, beta, alpha_bar)
        elapsed = time.time() - start_time
        avg_time = (elapsed / iterations) * 1000.0
    return avg_time

# A dummy model for benchmarking the sampling process
class DummyModel(nn.Module):
    def forward(self, x, t):
        return torch.randn_like(x)

def benchmark_sampling(update_func, model, scheduler, shape, device, iterations=10):
    for _ in range(2):
        _ = sample_ddpm(model, update_func, scheduler, shape, device=device)
    
    start_time = time.time()
    for _ in range(iterations):
        _ = sample_ddpm(model, update_func, scheduler, shape, device=device)
    elapsed = time.time() - start_time
    avg_ms = (elapsed / iterations) * 1000.0
    return avg_ms

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shape = (3, 1024, 1024)
    x = torch.randn(shape, device=device)
    eps = torch.randn(shape, device=device)
    
    alpha = 0.9
    beta = 0.1
    alpha_bar = 0.5

    iterations = 1000

    normal_time = benchmark_update(normal_update, x, eps, alpha, beta, alpha_bar, iterations)
    triton_time = benchmark_update(ddpm_kernel_update, x, eps, alpha, beta, alpha_bar, iterations)

    print(f"Average update time (normal PyTorch): {normal_time:.4f} ms")
    print(f"Average update time (Triton kernel): {triton_time:.4f} ms")
    
    T = 100
    scheduler = get_scheduler(T, beta_start=0.0001, beta_end=0.02, device=device)
    
    dummy_model = DummyModel().to(device)
    
    normal_sample_time = benchmark_sampling(normal_update, dummy_model, scheduler, shape, device, iterations=10)
    triton_sample_time = benchmark_sampling(ddpm_kernel_update, dummy_model, scheduler, shape, device, iterations=10)
    
    print(f"Average DDPM sampling time (normal PyTorch): {normal_sample_time:.4f} ms")
    print(f"Average DDPM sampling time (Triton kernel): {triton_sample_time:.4f} ms")
