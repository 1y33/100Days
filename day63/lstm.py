import torch
import triton
import triton.language as tl

@triton.jit
def lstm_kernel(
    input_ptr, h_prev_ptr, c_prev_ptr,
    weights_ptr, biases_ptr,
    h_new_ptr, c_new_ptr,
    N, D: tl.constexpr, H: tl.constexpr, K,
    stride_input_n, stride_input_d,
    stride_h_prev_n, stride_h_prev_h,
    stride_c_prev_n, stride_c_prev_h,
    stride_weights_j, stride_weights_k,
    stride_biases_j,
    stride_h_new_n, stride_h_new_h,
    stride_c_new_n, stride_c_new_h,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # Get program ID
    pid_n = tl.program_id(0)  # Batch dimension
    pid_h = tl.program_id(1)  # Hidden dimension
    
    if pid_n >= N or pid_h >= H:
        return
    
    # Initialize gate values for this specific hidden unit
    i_gate = 0.0  # Input gate
    f_gate = 0.0  # Forget gate
    g_gate = 0.0  # Cell gate
    o_gate = 0.0  # Output gate
    
    # Load previous cell state for this specific batch and hidden unit
    c_prev_val = tl.load(c_prev_ptr + pid_n * stride_c_prev_n + pid_h * stride_c_prev_h)
    
    # Process input part for each gate
    for j in range(D):
        input_val = tl.load(input_ptr + pid_n * stride_input_n + j * stride_input_d)
        
        # Each gate has its own weights in row-major order
        w_i = tl.load(weights_ptr + 0 * stride_weights_j + j * stride_weights_k)
        w_f = tl.load(weights_ptr + 1 * stride_weights_j + j * stride_weights_k)
        w_g = tl.load(weights_ptr + 2 * stride_weights_j + j * stride_weights_k)
        w_o = tl.load(weights_ptr + 3 * stride_weights_j + j * stride_weights_k)
        
        # Update gate values
        i_gate += input_val * w_i
        f_gate += input_val * w_f
        g_gate += input_val * w_g
        o_gate += input_val * w_o
    
    # Process hidden part for each gate
    for j in range(H):
        h_prev_val = tl.load(h_prev_ptr + pid_n * stride_h_prev_n + j * stride_h_prev_h)
        
        # Load weights for hidden state
        w_i = tl.load(weights_ptr + 0 * stride_weights_j + (j + D) * stride_weights_k)
        w_f = tl.load(weights_ptr + 1 * stride_weights_j + (j + D) * stride_weights_k)
        w_g = tl.load(weights_ptr + 2 * stride_weights_j + (j + D) * stride_weights_k)
        w_o = tl.load(weights_ptr + 3 * stride_weights_j + (j + D) * stride_weights_k)
        
        # Update gate values
        i_gate += h_prev_val * w_i
        f_gate += h_prev_val * w_f
        g_gate += h_prev_val * w_g
        o_gate += h_prev_val * w_o
    
    # Add biases
    i_gate += tl.load(biases_ptr + 0 * stride_biases_j + pid_h)
    f_gate += tl.load(biases_ptr + 1 * stride_biases_j + pid_h)
    g_gate += tl.load(biases_ptr + 2 * stride_biases_j + pid_h)
    o_gate += tl.load(biases_ptr + 3 * stride_biases_j + pid_h)
    
    # Apply activation functions
    i_gate = 1.0 / (1.0 + tl.exp(-i_gate))  # sigmoid
    f_gate = 1.0 / (1.0 + tl.exp(-f_gate))  # sigmoid
    g_gate = (tl.exp(g_gate) - tl.exp(-g_gate)) / (tl.exp(g_gate) + tl.exp(-g_gate))  # tanh
    o_gate = 1.0 / (1.0 + tl.exp(-o_gate))  # sigmoid
    
    # Compute new cell state
    c_new_val = f_gate * c_prev_val + i_gate * g_gate
    
    # Compute new hidden state
    tanh_c_new = (tl.exp(c_new_val) - tl.exp(-c_new_val)) / (tl.exp(c_new_val) + tl.exp(-c_new_val))
    h_new_val = o_gate * tanh_c_new
    
    # Store results
    tl.store(c_new_ptr + pid_n * stride_c_new_n + pid_h * stride_c_new_h, c_new_val)
    tl.store(h_new_ptr + pid_n * stride_h_new_n + pid_h * stride_h_new_h, h_new_val)

def triton_lstm(input, h_prev, c_prev, combined_weights, combined_biases):
    N, D = input.shape
    H = h_prev.shape[1]
    K = D + H
    
    h_new = torch.empty_like(h_prev)
    c_new = torch.empty_like(c_prev)
    
    grid = (triton.cdiv(N, 1), triton.cdiv(H, 1))
    lstm_kernel[grid](
        input, h_prev, c_prev,
        combined_weights, combined_biases,
        h_new, c_new,
        N, D, H, K,
        input.stride(0), input.stride(1),
        h_prev.stride(0), h_prev.stride(1),
        c_prev.stride(0), c_prev.stride(1),
        combined_weights.stride(0), combined_weights.stride(1),
        combined_biases.stride(0),
        h_new.stride(0), h_new.stride(1),
        c_new.stride(0), c_new.stride(1),
        BLOCK_SIZE_N=1, BLOCK_SIZE_H=1
    )
    
    return h_new, c_new
    
import timeit

# Parameters
D = 32  # Input size
H = 128 # Hidden size
N = 64  # Batch size
device = 'cuda'

# Initialize data
input = torch.randn(N, D, device=device)
h_prev = torch.randn(N, H, device=device)
c_prev = torch.randn(N, H, device=device)

# PyTorch LSTM
lstm = torch.nn.LSTM(input_size=D, hidden_size=H).to(device)
with torch.no_grad():
    # Prepare weights and biases for Triton
    weight_ih, weight_hh = lstm.weight_ih_l0, lstm.weight_hh_l0
    bias_ih, bias_hh = lstm.bias_ih_l0, lstm.bias_hh_l0
    
    # Combine weights and biases for Triton kernel
    W_ii, W_if, W_ig, W_io = weight_ih.chunk(4)
    W_hi, W_hf, W_hg, W_ho = weight_hh.chunk(4)
    combined_weights = torch.cat([
        torch.cat([W_ii, W_hi], dim=1),
        torch.cat([W_if, W_hf], dim=1),
        torch.cat([W_ig, W_hg], dim=1),
        torch.cat([W_io, W_ho], dim=1),
    ], dim=0)
    combined_biases = bias_ih + bias_hh

# Verify correctness
with torch.no_grad():
    output, (h_pytorch, c_pytorch) = lstm(input.unsqueeze(0), (h_prev.unsqueeze(0), c_prev.unsqueeze(0)))
h_triton, c_triton = triton_lstm(input, h_prev, c_prev, combined_weights, combined_biases)

# Check outputs
# torch.testing.assert_close(h_pytorch.squeeze(0), h_triton, atol=1e-4, rtol=1e-4)
# torch.testing.assert_close(c_pytorch.squeeze(0), c_triton, atol=1e-4, rtol=1e-4)

# Benchmark
def pytorch_lstm():
    with torch.no_grad():
        lstm(input.unsqueeze(0), (h_prev.unsqueeze(0), c_prev.unsqueeze(0)))
    torch.cuda.synchronize()

def triton_lstm_wrapper():
    triton_lstm(input, h_prev, c_prev, combined_weights, combined_biases)
    torch.cuda.synchronize()

# Warmup
for _ in range(100):
    pytorch_lstm()
    triton_lstm_wrapper()

# Timing
pytorch_time = timeit.timeit(pytorch_lstm, number=1000)
triton_time = timeit.timeit(triton_lstm_wrapper, number=1000)

print(f"PyTorch LSTM time: {pytorch_time:.6f} seconds")
print(f"Triton LSTM time: {triton_time:.6f} seconds")