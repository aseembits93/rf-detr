# GPU Profiling Guide for RF-DETR

This guide covers how to profile RF-DETR models using NVIDIA's profiling tools.

## Available Tools

This system is equipped with:

- **NVIDIA Driver**: 580.126.09 (CUDA 13.0 support)
- **CUDA Toolkit**: 12.8
- **GPU**: Tesla T4 (15GB memory)
- **Nsight Systems**: 2024.6.2 - Timeline profiling
- **Nsight Compute**: 12.8.1 - Kernel-level profiling
- **CUDA Profilers**: nvprof, nvvp (legacy tools)

## Quick Start

### Using the Helper Script

The easiest way to profile is using the provided `profile_gpu.sh` script:

```bash
# Profile training with Nsight Systems (default)
./profile_gpu.sh -- python -m rfdetr.train --config configs/rtdetr_r50vd_6x.yml

# Profile inference with Nsight Compute
./profile_gpu.sh -p ncu -- python benchmark_inference.py

# Run both profilers
./profile_gpu.sh -p both -n my_experiment -- python train.py

# Custom output directory
./profile_gpu.sh -o ./my_profiles -- python script.py
```

View help for all options:
```bash
./profile_gpu.sh --help
```

### Manual Profiling

If you prefer direct control:

**Nsight Systems (Timeline Profiling):**
```bash
nsys profile \
    --trace=cuda,nvtx,cudnn,cublas \
    --cudabacktrace=true \
    --python-sampling=true \
    --output=profile_output \
    python your_script.py
```

**Nsight Compute (Kernel Profiling):**
```bash
# Basic metrics (fast)
ncu --set basic --export=profile_output python your_script.py

# Full metrics (comprehensive but slow)
ncu --set full --export=profile_output python your_script.py

# Specific kernel only
ncu --kernel-name="your_kernel_name" --export=profile_output python your_script.py
```

## Profiling Workflows

### 1. Overall Performance Analysis (Use Nsight Systems)

Best for:
- Finding bottlenecks in the training/inference pipeline
- Understanding GPU utilization
- Analyzing data loading performance
- Identifying CPU-GPU synchronization issues

```bash
./profile_gpu.sh -p nsys -n training_profile -- \
    python -m rfdetr.train --config configs/rtdetr_r50vd_6x.yml --epochs 1
```

View results:
```bash
# Generate statistics report
nsys stats profile_results/training_profile_nsys.nsys-rep

# GUI viewer (requires X11 forwarding)
nsys-ui profile_results/training_profile_nsys.nsys-rep
```

### 2. Kernel Optimization (Use Nsight Compute)

Best for:
- Optimizing specific CUDA kernels
- Understanding memory bandwidth utilization
- Analyzing warp efficiency
- Detailed SM (Streaming Multiprocessor) metrics

```bash
# Profile specific kernels only (faster)
NCU_OPTS="--kernel-regex 'your_pattern' --set full" \
    ./profile_gpu.sh -p ncu -n kernel_profile -- python benchmark.py
```

View results:
```bash
# Command-line statistics
ncu --import profile_results/kernel_profile_ncu.ncu-rep --page raw

# GUI viewer (requires X11 forwarding)
ncu-ui profile_results/kernel_profile_ncu.ncu-rep
```

### 3. Memory Profiling

**Check memory usage:**
```bash
# Monitor GPU memory during execution
watch -n 0.5 nvidia-smi

# Or use Python torch utilities
python -c "import torch; print(torch.cuda.memory_summary())"
```

**Profile memory with Nsight Systems:**
```bash
NSYS_OPTS="--gpu-metrics-device=all" \
    ./profile_gpu.sh -p nsys -- python your_script.py
```

### 4. Multi-GPU Profiling

For distributed training:
```bash
# Profile all GPUs
nsys profile \
    --trace=cuda,nvtx,mpi \
    --mpi-impl=openmpi \
    --output=profile_rank_%q{SLURM_PROCID} \
    python -m torch.distributed.launch \
        --nproc_per_node=4 train.py
```

## Adding NVTX Annotations (Recommended)

To get better insights, add NVTX markers to your code:

```python
import torch

# Annotate code regions
with torch.cuda.nvtx.range("data_loading"):
    data = load_batch()

with torch.cuda.nvtx.range("forward_pass"):
    output = model(data)

with torch.cuda.nvtx.range("backward_pass"):
    loss.backward()
```

These annotations will show up as labeled regions in Nsight Systems timeline.

## Interpreting Results

### Nsight Systems Key Metrics

1. **GPU Utilization**: Should be >80% during training
2. **CUDA API Calls**: Look for excessive synchronization
3. **Python Function Timeline**: Identify slow Python operations
4. **Memory Transfers**: Minimize H2D (Host-to-Device) and D2H transfers

### Nsight Compute Key Metrics

1. **SM Efficiency**: How well the kernel uses streaming multiprocessors
2. **Memory Bandwidth**: Compare achieved vs. theoretical bandwidth
3. **Warp Efficiency**: Measure of thread divergence
4. **Occupancy**: How well the kernel utilizes GPU resources

### Common Issues and Solutions

**Low GPU Utilization:**
- Increase batch size
- Reduce data loading time (more workers, prefetching)
- Minimize CPU-GPU synchronization

**High Memory Transfers:**
- Pin memory: `DataLoader(..., pin_memory=True)`
- Use `.to(device, non_blocking=True)`
- Reduce data transfer frequency

**Kernel Launch Overhead:**
- Fuse operations where possible
- Use compiled models (torch.compile)
- Reduce number of small kernels

## Environment Variables

Control profiling behavior:

```bash
# Enable CUDA launch blocking (for debugging)
export CUDA_LAUNCH_BLOCKING=1

# Enable cuDNN benchmarking (may affect profiling)
export TORCH_CUDNN_BENCHMARK=1

# Disable PyTorch profiler overhead
export TORCH_PROFILER_ENABLE=0

# Set CUPTI buffer size (for large profiles)
export CUPTI_BUFFER_SIZE=32
```

## Benchmarking Examples

### Inference Benchmark

```python
import torch
import time
from rfdetr import RTDETR

model = RTDETR("rtdetr-l").cuda().eval()
dummy_input = torch.randn(1, 3, 640, 640).cuda()

# Warmup
for _ in range(10):
    _ = model(dummy_input)

torch.cuda.synchronize()

# Benchmark
iterations = 100
start = time.time()
for _ in range(iterations):
    _ = model(dummy_input)
torch.cuda.synchronize()
end = time.time()

print(f"Average inference time: {(end - start) / iterations * 1000:.2f} ms")
```

Profile this with:
```bash
./profile_gpu.sh -n inference_benchmark -- python benchmark_inference.py
```

### Training Benchmark

```python
# Profile one training step
import torch
from rfdetr import RTDETR

model = RTDETR("rtdetr-l").cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

dummy_images = torch.randn(4, 3, 640, 640).cuda()
dummy_targets = [
    {"boxes": torch.rand(10, 4).cuda(), "labels": torch.randint(0, 80, (10,)).cuda()}
    for _ in range(4)
]

# Profile forward + backward
torch.cuda.synchronize()
with torch.cuda.nvtx.range("training_step"):
    outputs = model(dummy_images)
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
```

## CI/CD Integration

For automated performance testing:

```bash
# Run lightweight profiling in CI
if [ -n "$PROFILE_GPU" ]; then
    ./profile_gpu.sh -p nsys -n ci_profile -- \
        pytest tests/ -m gpu --profile-svg
fi
```

## Troubleshooting

**"Permission denied" errors:**
```bash
# Check permissions
ls -la /usr/local/cuda-12.8/bin/ncu
ls -la /usr/local/bin/nsys

# Ensure CUDA paths are in environment
source ~/.bashrc
```

**"CUPTI initialization failed":**
```bash
# Check CUPTI library
ldconfig -p | grep cupti

# Add to LD_LIBRARY_PATH if needed
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

**Profile files too large:**
```bash
# Limit profiling duration
nsys profile --duration=30 --output=profile python script.py

# Profile specific kernels only
ncu --kernel-regex "conv|gemm" --export=profile python script.py
```

**GUI tools not working:**
```bash
# Enable X11 forwarding (SSH)
ssh -X user@host

# Or export profile and view locally
scp remote:profile.nsys-rep ./
nsys-ui profile.nsys-rep
```

## Additional Resources

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Profiling Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [PyTorch Profiling Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

## Quick Reference

```bash
# Check GPU status
nvidia-smi

# Check CUDA version
nvcc --version
cat /usr/local/cuda/version.txt

# Check profiler versions
nsys --version
ncu --version

# View existing profile
nsys stats profile.nsys-rep
ncu --import profile.ncu-rep --page raw

# Convert profile to other formats
nsys export --type=sqlite profile.nsys-rep
```
