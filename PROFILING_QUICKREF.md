# GPU Profiling Quick Reference

## System Info
- **GPU**: Tesla T4 (15GB)
- **Driver**: 580.126.09 (CUDA 13.0)
- **CUDA Toolkit**: 12.8
- **Nsight Systems**: 2024.6.2
- **Nsight Compute**: 12.8.1
- **nsight-python**: 0.9.6

## Quick Commands

### Using the Helper Script (Recommended)

```bash
# Basic profiling (Nsight Systems)
./profile_gpu.sh -- python your_script.py

# Kernel profiling (Nsight Compute - slower)
./profile_gpu.sh -p ncu -- python your_script.py

# Run both profilers
./profile_gpu.sh -p both -- python your_script.py

# Custom output
./profile_gpu.sh -n my_experiment -o ./my_results -- python script.py

# Help
./profile_gpu.sh --help
```

### Direct Commands

**Nsight Systems (Timeline):**
```bash
nsys profile --trace=cuda,nvtx --output=profile python script.py
nsys stats profile.nsys-rep  # View stats
```

**Nsight Compute (Kernels):**
```bash
ncu --set basic --export=profile python script.py
ncu --import profile.ncu-rep --page raw  # View stats
```

**Monitor GPU:**
```bash
watch -n 0.5 nvidia-smi
```

**Analyze Profiles:**
```bash
# Automated analysis with nsight-python
python profile_nsight_analysis.py profile.nsys-rep

# Summary only (fast)
python profile_nsight_analysis.py profile.nsys-rep --summary-only

# Full analysis with extraction APIs
python profile_nsight_analysis.py profile.nsys-rep --full
```

## Python Profiling Utilities

The `profile_utils.py` module provides helpers:

```python
from profile_utils import nvtx_range, CUDATimer, benchmark_inference, profile_section

# Add NVTX markers (shows in Nsight Systems)
with nvtx_range("my_operation"):
    result = model(input)

# Accurate CUDA timing
with CUDATimer() as timer:
    output = model(input)
print(f"Time: {timer.elapsed_time_ms:.2f} ms")

# Profile with timing + memory
with profile_section("training_step"):
    loss.backward()

# Benchmark inference
stats = benchmark_inference(model, input, num_iterations=100)
print(f"Mean: {stats['mean_ms']:.2f} ms")
```

## Common Workflows

### 1. Find Bottlenecks
```bash
./profile_gpu.sh -n bottleneck_check -- \
    uv run python -m rfdetr.train --config configs/rtdetr_r50vd_6x.yml --epochs 1
```

### 2. Benchmark Inference
```python
# Create benchmark script
import torch
from rfdetr import RTDETR
from profile_utils import benchmark_inference

model = RTDETR("rtdetr-l").cuda().eval()
input = torch.randn(1, 3, 640, 640).cuda()
stats = benchmark_inference(model, input)
print(f"Inference: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
```

Profile it:
```bash
./profile_gpu.sh -n inference_bench -- uv run python benchmark.py
```

### 3. Memory Profiling
```python
from profile_utils import print_memory_usage

model.cuda()
print_memory_usage("After model load: ")

output = model(input)
print_memory_usage("After forward: ")
```

### 4. Profile Specific Code Section
```python
import torch
from profile_utils import profile_section

with profile_section("data_loading"):
    data = dataset[idx]

with profile_section("forward_pass"):
    output = model(data)
```

## Interpreting Results

### Nsight Systems
Look for:
- **GPU Utilization**: Should be >80% during compute
- **Memory Transfers**: Minimize H2D/D2H
- **Gaps**: Indicates CPU-GPU sync issues
- **NVTX Ranges**: Your custom markers

### Nsight Compute  
Key metrics:
- **SM Efficiency**: >80% is good
- **Memory Bandwidth**: Compare to theoretical max
- **Occupancy**: Higher is generally better
- **Warp Efficiency**: Watch for divergence

## Optimization Tips

**Low GPU Utilization:**
- Increase batch size
- More DataLoader workers
- Enable `pin_memory=True`

**Slow Data Loading:**
- More workers: `DataLoader(num_workers=4)`
- Prefetch factor: `DataLoader(prefetch_factor=2)`
- Use `pin_memory=True`

**High Memory:**
- Reduce batch size
- Use gradient checkpointing
- Enable mixed precision (AMP)

**Slow Kernels:**
- Use `torch.compile()` (PyTorch 2.0+)
- Check for unnecessary `.cpu()/.cuda()` calls
- Profile specific kernels with `ncu`

## Environment Variables

```bash
# Enable CUDA launch blocking (debugging)
export CUDA_LAUNCH_BLOCKING=1

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0

# CUPTI buffer size (for large profiles)
export CUPTI_BUFFER_SIZE=32
```

## Common Issues

**Permission denied:**
```bash
chmod +x profile_gpu.sh
source ~/.bashrc  # Ensure CUDA in PATH
```

**Profile too large:**
```bash
# Limit duration
nsys profile --duration=30 --output=profile python script.py

# Profile specific kernels
ncu --kernel-regex "conv|gemm" python script.py
```

**CUPTI errors:**
```bash
# Check CUPTI
ldconfig -p | grep cupti

# Add to environment
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

## View Results

**Command-line:**
```bash
# Nsight Systems
nsys stats profile.nsys-rep --report cuda_api_sum
nsys stats profile.nsys-rep --report cuda_gpu_kern_sum

# Nsight Compute
ncu --import profile.ncu-rep --page raw
```

**GUI (requires X11):**
```bash
nsys-ui profile.nsys-rep
ncu-ui profile.ncu-rep
```

**Remote viewing:**
```bash
# Download profile
scp user@remote:profile.nsys-rep ./

# Open locally
nsys-ui profile.nsys-rep
```

## Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `profile_gpu.sh` | Easy profiling wrapper | `./profile_gpu.sh -- python script.py` |
| `profile_utils.py` | NVTX annotations, timing | `from profile_utils import nvtx_range` |
| `profile_nsight.py` | Nsight-python profiling | `nsys profile ... python profile_nsight.py` |
| `profile_nsight_analysis.py` | Analyze profiles | `python profile_nsight_analysis.py profile.nsys-rep` |

## More Info

- **Detailed guide**: `PROFILING.md`
- **Nsight Python guide**: `PROFILING_NSIGHT_PYTHON.md`
- **This quick reference**: `PROFILING_QUICKREF.md`
