# Project Progress and Tasks

Bro in CUDA 📗 : https://github.com/a-hamdi/cuda

Mentor 🚀 : https://github.com/hkproj | https://github.com/hkproj/100-days-of-gpu
### Mandatory and Optional Tasks
| Day   | Task Description                                                                                     |
|-------|-----------------------------------------------------------------------------------------------------|
| D15   | **Mandatory FA2-Forward**: Implement forward pass for FA2 (e.g., a custom neural network layer).    |
| D20   | **Mandatory FA2-Backwards**: Implement backward pass for FA2 (e.g., gradient computation).          |
| D20   | **Optional Fused Chunked CE Loss + Backwards**: Fused implementation of chunked cross-entropy loss with backward pass. Can use Liger Kernel as a reference implementation. |

---

### Project Progress by Day
| Day   | Files & Summaries                                                                                                                                                                                                                          |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| day1  | **printAdd.cu**: Print global indices for 1D vector (index calculation).<br>**addition.cu**: GPU vector addition; basics of memory allocation/host-device transfer.                                                                 |
| day2  | **function.cu**: Use `__device__` function in kernel; per-thread calculations.                                                                                                                                                       |
| day3  | **addMatrix.cu**: 2D matrix addition; map row/column indices to threads.<br>**anotherMatrix.cu**: Transform matrices with custom function; 2D index operations.                                                                       |
| day4  | **layerNorm.cu**: Layer normalization using shared memory; mean/variance computation.                                                                                                                                                |
| day5  | **vectorSumTricks.cu**: Parallel vector sum via reduction; shared memory optimizations.                                                                                                                                               |
| day6  | **SMBlocks.cu**: Retrieve SM ID per thread via inline PTX.<br>**SoftMax.cu**: Shared-memory softmax; split exponent/normalization steps.<br>**TransposeMatrix.cu**: Matrix transpose via index swapping.<br>**ImportingToPython/rollcall.cu**: Python-CUDA integration.<br>**AdditionKernel/additionKernel.cu**: Modify PyTorch tensors in CUDA. |
| day7  | **naive.cu**: Naive matrix multiplication.<br>**matmul.cu**: Tiled matmul with shared memory.<br>**conv1d.cu**: 1D convolution with shared memory.<br>**pythontest.py**: Validate custom convolution against PyTorch.                               |
| day8  | **pmpbook/chapter3matvecmul.cu**: Matrix-vector multiplication.<br>**pmpbook/chapter3ex.cu**: Benchmarks different matrix add kernels.<br>**pmpbook/deviceinfo.cu**: Prints device properties.<br>**pmpbook/color2gray.cu**: Convert RGB to grayscale.<br>**pmpbook/vecaddition.cu**: Another vector addition example.<br>**pmpbook/imageblur.cu**: Simple image blur.<br>**selfAttention/selfAttention.cu**: Self-attention kernel with online softmax. |
| day9  | **flashAttentionFromTut.cu**: Minimal Flash Attention kernel with shared memory tiling.<br>**bind.cpp**: Torch C++ extension bindings for Flash Attention.<br>**test.py**: Tests the minimal Flash Attention kernel against a manual softmax-based attention for comparison. |
| day10 | **ppmbook/matrixmul.cu**: Matrix multiplication using CUDA.<br>**setup.py**: Torch extension build script for CUDA code (FlashAttention).<br>**FlashAttention.cu**: Example Flash Attention CUDA kernel.<br>**FlashAttention.cpp**: Torch bindings for the Flash Attention kernel.<br>**test.py**: Manual vs. CUDA-based attention test.<br>**linking/test.py**: Builds simple CUDA kernel for testing linking.<br>**linking/simpleKernel.cpp**: Torch extension binding for a simple CUDA kernel.<br>**linking/simpleKernel.cu**: Simple CUDA kernel that increments a tensor. |
| day11 | **FlashTestPytorch/**: Custom Flash Attention in PyTorch, tests and benchmarks.<br>**testbackward.py**: Gradient comparison between custom CUDA kernels and PyTorch. |
| day12 | **softMax.cu**: Additional softmax kernel with shared memory optimization.<br>**NN/kernels.cu**: Tiled kernel implementation and layer initialization.<br>**tileMatrix.cu**: Demonstrates tile-based matrix operations. |
| day13 | **RMS.cu**: RMS kernel (V1) with naive sum-of-squares approach.<br>**RMSBetter.cu**: RMS kernel (V2) using warp-reduce optimization,float4 +others .<br>**binding.cpp**: Torch bindings for RMS kernels.<br>**test.py**: Tests and benchmarks RMS kernels vs PyTorch. |
| day14 | **FA2/flash.cu & kernels.cu**: Second iteration of Flash Attention featuring partial forward/backward logic. <br>**helper.cuh**: Utility functions and warp-reduce helpers. <br>**conv.cu**: Basic 2D convolution with shared memory. |
| day15 | **Attention.cu**: Single-headed attention kernel vs. CPU reference. <br>**dotproduct.cu**: Batched/tiled dot-product kernel for vectors or matrices. <br>**SMM.cu**: Sparse matrix multiplication in CSR format. |
| day16 | **attentionbwkd.cu**: Extends attention with gradient computation; forward & backward passes. |
| day17 | **cublas1.cu**, **cublas2.cu**, **cublas3.cu**: Various cuBLAS examples for dot products, axpy, max/min, and other BLAS operations. |
| day18 | **wrap.cu**: Warp-based reduction and max-finding with inline PTX.<br>**atomic1.cu**, **atomic2.cu**: Implement and test custom atomic increment operations. |
| day19 | **cublasMM.cu**: Matrix multiplication with cuBLAS plus a simple self-attention example. |
| day20 | **rope.cu**: Rope (rotary positional encoding) kernel and its PyTorch extension.<br>**test_rope.py**: Benchmarks for the rope kernel. |

#### How to load into Pytorch:
- (optional) create tempalte kernel
- create kernelforward where you set up the grids and other calculations
- create `.cpp` file
- import the header of the file
- create a wraper so that you can use tensors
- use PYBIN11_MODULE to create a torchextension
- in `.py` file : `torch.utils.cpp_extension.load()` use it to load the files and it will compile