# Nsight Python SDK Profiling Guide

This guide covers using the `nsight-python` package for automated profiling and analysis.

## Overview

The `nsight-python` package provides:
- **NVTX Annotations**: Programmatic region marking (non-nested)
- **Profile Analysis**: Extract and analyze data from `.nsys-rep` files
- **Automation**: Python API for profiling workflows

## Installation

```bash
uv sync --group profiling
```

## Scripts

### 1. `profile_nsight.py` - Profiling with NVTX Annotations

This script demonstrates using NVTX markers in your code. The annotations will appear in Nsight Systems profiles when you run the code under `nsys profile`.

**Usage:**

```bash
# Profile with benchmark mode
nsys profile --trace=cuda,nvtx --output=profile_results/benchmark \
    uv run python profile_nsight.py --mode benchmark --model-type simple

# Profile with RF-DETR model
nsys profile --trace=cuda,nvtx --output=profile_results/rfdetr_inference \
    uv run python profile_nsight.py --mode benchmark --model-type rfdetr --iterations 10

# Profile training step
nsys profile --trace=cuda,nvtx --output=profile_results/training \
    uv run python profile_nsight.py --mode training --model-type simple
```

**Options:**
```bash
--mode {benchmark,training}    # Profiling mode
--model-type {simple,rfdetr}   # Model to profile
--iterations N                 # Number of benchmark iterations
--warmup N                     # Warmup iterations
--batch-size N                 # Batch size
```

### 2. `profile_nsight_analysis.py` - Analyze Existing Profiles

Analyze `.nsys-rep` files using the nsight-python extraction APIs.

**Usage:**

```bash
# Generate summary report
python profile_nsight_analysis.py profile_results/my_profile.nsys-rep

# Generate summary only (fast)
python profile_nsight_analysis.py profile.nsys-rep --summary-only

# Full analysis with extraction APIs (slower)
python profile_nsight_analysis.py profile.nsys-rep --full

# Export to SQLite only
python profile_nsight_analysis.py profile.nsys-rep --export-sqlite
```

## Using NVTX Annotations in Your Code

### Method 1: Direct torch.cuda.nvtx (Supports Nesting)

```python
import torch

# Nested annotations work fine
torch.cuda.nvtx.range_push("outer_region")
torch.cuda.nvtx.range_push("inner_region")
# ... code ...
torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_pop()

# Context manager (from profile_utils.py)
from profile_utils import nvtx_range

with nvtx_range("my_operation"):
    result = model(input)
```

### Method 2: nsight-python annotation (No Nesting)

```python
from nsight import annotation

# Single-level annotations only (no nesting)
with annotation.annotate("operation_1"):
    # ... code ...
    pass

# This will ERROR (nested not supported):
# with annotation.annotate("outer"):
#     with annotation.annotate("inner"):  # ValueError!
#         pass
```

**Recommendation:** Use `torch.cuda.nvtx` or `profile_utils.nvtx_range()` for most cases since they support nesting.

## Complete Workflow Example

### 1. Add Annotations to Your Code

```python
import torch
from profile_utils import nvtx_range, CUDATimer

def train_step(model, data, optimizer):
    with nvtx_range("training_step"):
        # Data loading
        with nvtx_range("data_prep"):
            images, targets = data

        # Forward
        with nvtx_range("forward"):
            outputs = model(images, targets)
            loss = outputs["loss"]

        # Backward
        with nvtx_range("backward"):
            loss.backward()

        # Optimizer
        with nvtx_range("optimizer"):
            optimizer.step()
            optimizer.zero_grad()

    return loss.item()
```

### 2. Profile the Code

```bash
# Using helper script
./profile_gpu.sh -n my_training -- python train.py

# Or manually with nsys
nsys profile \
    --trace=cuda,nvtx,cudnn \
    --output=profile_results/training \
    python train.py
```

### 3. Analyze the Profile

```bash
# Quick summary
python profile_nsight_analysis.py profile_results/training.nsys-rep

# Detailed analysis
python profile_nsight_analysis.py profile_results/training.nsys-rep --full

# View in GUI
nsys-ui profile_results/training.nsys-rep
```

## Example: RF-DETR Inference Profiling

```bash
# Step 1: Create a benchmark script with annotations
cat > benchmark_rfdetr.py << 'EOF'
import torch
from rfdetr import RTDETR
from profile_utils import nvtx_range, benchmark_inference

model = RTDETR("rtdetr-l").cuda().eval()
input_tensor = torch.randn(1, 3, 640, 640).cuda()

# Annotate the benchmark
with nvtx_range("rfdetr_benchmark"):
    stats = benchmark_inference(model, input_tensor, num_iterations=100)

print(f"Mean inference time: {stats['mean_ms']:.2f} ms")
EOF

# Step 2: Profile it
./profile_gpu.sh -n rfdetr_benchmark -- python benchmark_rfdetr.py

# Step 3: Analyze
python profile_nsight_analysis.py profile_results/rfdetr_benchmark.nsys-rep --summary-only
```

## Extracting Data with nsight-python

For advanced users who want to programmatically analyze profiles:

```python
from nsight import extraction
import pandas as pd

# Export profile to SQLite first
# $ nsys export --type=sqlite profile.nsys-rep

# Extract CUDA API calls
api_calls = extraction.get_cuda_api_calls("profile.sqlite")
df = api_calls.to_pandas()
print(df.head())

# Extract kernel launches
kernels = extraction.get_cuda_kernels("profile.sqlite")
kernel_df = kernels.to_pandas()
print(kernel_df.groupby("shortName")["duration"].sum())

# Extract memory operations
mem_ops = extraction.get_memory_operations("profile.sqlite")
mem_df = mem_ops.to_pandas()
print(f"Total data transferred: {mem_df['bytes'].sum() / 1024**2:.2f} MB")
```

**Note:** The extraction APIs require:
1. Profile exported to SQLite format
2. Specific nsight-python version (some features may vary)
3. Optional: pandas for DataFrame conversion

## Comparison: Different Profiling Approaches

### 1. `profile_gpu.sh` (Shell Script)
- ✅ Easy to use
- ✅ Works with any Python script
- ✅ No code modifications needed
- ❌ Basic NVTX only (no custom annotations)

### 2. `profile_utils.py` (Python Utilities)
- ✅ Add custom NVTX markers
- ✅ Accurate CUDA timing
- ✅ Memory profiling
- ✅ Works in any profiling context
- ❌ Requires code modifications

### 3. `profile_nsight.py` (Nsight Python SDK)
- ✅ Automated profiling workflow
- ✅ NVTX annotations for benchmarks
- ❌ Must run under `nsys profile`
- ❌ No nested annotations with `annotation.annotate()`

### 4. `profile_nsight_analysis.py` (Analysis Tool)
- ✅ Automated analysis of profiles
- ✅ Extract metrics programmatically
- ✅ Generate reports
- ❌ Requires existing `.nsys-rep` file

## Best Practices

1. **For Quick Profiling**: Use `profile_gpu.sh`
   ```bash
   ./profile_gpu.sh -- python your_script.py
   ```

2. **For Annotated Profiling**: Use `profile_utils.py` + `profile_gpu.sh`
   ```python
   from profile_utils import nvtx_range
   with nvtx_range("my_section"):
       # code
   ```
   ```bash
   ./profile_gpu.sh -- python annotated_script.py
   ```

3. **For Automated Analysis**: Use `profile_nsight_analysis.py`
   ```bash
   python profile_nsight_analysis.py profile.nsys-rep --full
   ```

4. **Always Profile on GPU**: NVTX annotations only work with CUDA code

5. **Use Descriptive Names**: Make annotation names clear and specific
   ```python
   # Good
   with nvtx_range("resnet50_forward"):
   
   # Bad
   with nvtx_range("model"):
   ```

## Troubleshooting

**"Nested annotations are not supported"**
- Use `torch.cuda.nvtx` instead of `nsight.annotation.annotate`
- Or use `profile_utils.nvtx_range()`

**"No NVTX data in profile"**
- Make sure to use `--trace=nvtx` when running nsys
- Check that CUDA is available and being used

**"Module 'nsight' not found"**
- Run: `uv sync --group profiling`

**Extraction APIs not working**
- Export profile to SQLite first: `nsys export --type=sqlite profile.nsys-rep`
- Check nsight-python version compatibility
- Some features may vary by version

## Additional Resources

- Main profiling guide: `PROFILING.md`
- Quick reference: `PROFILING_QUICKREF.md`
- Profile utilities: `profile_utils.py`
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [NVTX Documentation](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx)
