# CUDA Streams Explained

## What Are CUDA Streams?

**CUDA streams are sequences of GPU operations that execute in order.** Think of them as independent "queues" or "lanes" of work on the GPU.

### Simple Analogy

Imagine a highway with multiple lanes:
- **Single stream** = One lane highway (cars must go one after another)
- **Multiple streams** = Multi-lane highway (cars in different lanes can drive simultaneously)

Each "car" is a GPU operation (kernel, memory copy, etc.)

---

## Key Concepts

### 1. Sequential Execution Within a Stream

Operations in the **same stream** execute in order:

```
Stream 0:  [Kernel A] → [Kernel B] → [Kernel C]
           ↓          ↓          ↓
           Must wait  Must wait  Done
```

- Kernel B waits for Kernel A to finish
- Kernel C waits for Kernel B to finish

### 2. Concurrent Execution Across Streams

Operations in **different streams** can overlap:

```
Stream 0:  [Kernel A] → [Kernel B] → [Kernel C]
Stream 1:  [Kernel D] → [Kernel E] → [Kernel F]
           ↓          ↓          ↓
           Can run    Can run    Can run
           at same    at same    at same
           time       time       time
```

- Kernel A and Kernel D can execute simultaneously
- Kernel B and Kernel E can overlap
- Limited by GPU resources (SMs, memory bandwidth)

---

## Why Use CUDA Streams?

### 1. Hide Latency

**Problem:** Memory transfers are slow
```
Without streams:
[Copy H→D] → [Kernel] → [Copy D→H]
^^^^^^^^^    ^^^^^^^^    ^^^^^^^^^^
100ms        50ms        100ms
Total: 250ms
```

**Solution:** Overlap transfers with computation
```
With streams:
Stream 0: [Copy H→D Batch 1] → [Kernel Batch 1] → [Copy D→H Batch 1]
Stream 1:           [Copy H→D Batch 2] → [Kernel Batch 2] → [Copy D→H Batch 2]
Stream 2:                       [Copy H→D Batch 3] → [Kernel Batch 3] → ...

Total: Much faster! (overlapped execution)
```

### 2. Increase GPU Utilization

**Without streams:**
```
GPU Usage: ▓▓▓▓░░░░▓▓▓▓░░░░▓▓▓▓  (50% utilization)
           Kernel  Wait  Kernel  Wait  Kernel
```

**With streams:**
```
GPU Usage: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (100% utilization)
           Multiple kernels running concurrently
```

### 3. Reduce CPU-GPU Synchronization Overhead

**Benefit:** CPU can submit work to multiple streams without waiting
```
CPU:  Submit→Submit→Submit→ (continue working)
         ↓      ↓      ↓
GPU:  [S0]   [S1]   [S2]   (execute in parallel)
```

---

## CUDA Streams in RF-DETR

### From Our Nsight Profile

Looking at the profiling data, we saw operations on different streams:

```
Stream 7:  [CUDA memcpy D→H]
Stream 13: [CUDA memcpy H→D]
Stream 15: [CUDA memset]
Stream 19: [CUDA memcpy D→H]
Stream 20: [CUDA memcpy D→H]
Stream 21: [CUDA memcpy D→H]
Stream 22: [GPU Kernels]      ← Main computation stream
Stream 26: [GPU Kernels]      ← Additional stream
```

**What's happening:**
- TensorRT automatically uses multiple streams
- Stream 22: Main inference kernels
- Stream 26: Postprocessing operations
- Streams 7, 13, 15, 19-21: Memory operations

**Why TensorRT does this:**
- Overlap memory transfers with computation
- Run independent operations in parallel
- Maximize GPU utilization

---

## Types of Streams

### 1. Default Stream (Stream 0)

```python
import torch

# Implicit default stream
tensor = torch.randn(1000, 1000).cuda()
result = tensor @ tensor  # Runs on default stream
```

**Characteristics:**
- Synchronizes with all other streams
- Blocks until all operations complete
- Simple but can be slow

### 2. Non-Blocking Streams

```python
import torch

# Create custom stream
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    # Operations on this stream
    tensor = torch.randn(1000, 1000).cuda()
    result = tensor @ tensor

# Can continue on CPU immediately
print("CPU continues while GPU works")
```

**Characteristics:**
- Don't synchronize with other streams
- Can run concurrently
- More complex but faster

---

## Practical Examples

### Example 1: Sequential Execution (Slow)

```python
import torch
import time

def sequential_inference(model, images):
    """Process images one at a time."""
    results = []
    
    for img in images:
        # Copy to GPU
        gpu_img = img.cuda()  # Blocks until complete
        
        # Run inference
        output = model(gpu_img)  # Blocks until complete
        
        # Copy back to CPU
        result = output.cpu()  # Blocks until complete
        results.append(result)
    
    return results

# Time: N × (copy_time + inference_time + copy_time)
```

**Timeline:**
```
Image 1: [H→D] → [Kernel] → [D→H]
Image 2:              [H→D] → [Kernel] → [D→H]
Image 3:                           [H→D] → [Kernel] → [D→H]

Total time: 3 × (H→D + Kernel + D→H)
```

### Example 2: Concurrent Execution with Streams (Fast)

```python
import torch

def concurrent_inference(model, images):
    """Process multiple images concurrently using streams."""
    num_streams = 4
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    
    # Allocate pinned memory for fast transfers
    pinned_inputs = [torch.cuda.pin_memory(img) for img in images]
    
    results = []
    
    for i, (img, stream) in enumerate(zip(pinned_inputs, streams)):
        with torch.cuda.stream(stream):
            # Copy to GPU (async)
            gpu_img = img.cuda(non_blocking=True)
            
            # Run inference (async)
            output = model(gpu_img)
            
            # Copy back (async)
            result = output.cpu()
            results.append(result)
    
    # Synchronize all streams at the end
    for stream in streams:
        stream.synchronize()
    
    return results

# Time: max(H→D, Kernel, D→H) + overhead
# Much faster! Operations overlap
```

**Timeline:**
```
Stream 0: [H→D Img1] → [Kernel Img1] → [D→H Img1]
Stream 1:   [H→D Img2] → [Kernel Img2] → [D→H Img2]
Stream 2:     [H→D Img3] → [Kernel Img3] → [D→H Img3]
Stream 3:       [H→D Img4] → [Kernel Img4] → [D→H Img4]

Total time: ≈ (H→D + Kernel + D→H) + small overhead
Speedup: Nearly 4x!
```

---

## Stream Synchronization

### The Problem

Streams execute asynchronously, so CPU doesn't wait:

```python
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    result = model(input)  # Started on GPU

print(result)  # ❌ ERROR! Result not ready yet
```

### Solutions

**1. Explicit Synchronization**
```python
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    result = model(input)

stream.synchronize()  # ✓ Wait for stream to finish
print(result)  # Now safe
```

**2. Event-Based Synchronization**
```python
stream = torch.cuda.Stream()
event = torch.cuda.Event()

with torch.cuda.stream(stream):
    result = model(input)
    event.record()  # Mark this point in stream

# Do other work on CPU...

event.synchronize()  # Wait for event
print(result)  # Now safe
```

**3. Stream Wait Stream**
```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    intermediate = model_part1(input)

# Make stream2 wait for stream1
event = torch.cuda.Event()
event.record(stream1)
stream2.wait_event(event)

with torch.cuda.stream(stream2):
    output = model_part2(intermediate)
```

---

## Real-World Example: RF-DETR Pipeline

### Without Streams (Sequential)

```python
def inference_sequential(model, images):
    """Current approach - sequential processing."""
    results = []
    
    for img in images:
        # Everything blocks
        result = model.infer(img)  # 12.58ms per image
        results.append(result)
    
    return results

# Total: 4 images × 12.58ms = 50.32ms
```

### With Streams (Concurrent)

```python
def inference_concurrent(model, images):
    """Optimized approach - concurrent processing."""
    num_streams = 4
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    results = [None] * len(images)
    
    # Process in batches of num_streams
    for batch_start in range(0, len(images), num_streams):
        batch_end = min(batch_start + num_streams, len(images))
        
        # Submit work to all streams
        for i in range(batch_start, batch_end):
            stream_idx = i % num_streams
            with torch.cuda.stream(streams[stream_idx]):
                results[i] = model.infer(images[i])
        
        # Wait for batch to complete
        for stream in streams:
            stream.synchronize()
    
    return results

# Total: ~15-20ms for 4 images (2-3x speedup!)
```

---

## Streams in TensorRT

TensorRT automatically uses streams for optimization:

```cpp
// TensorRT internal (simplified)
class ExecutionContext {
    cudaStream_t main_stream;
    cudaStream_t memory_stream;
    cudaStream_t postprocess_stream;
    
    void execute() {
        // Stream 1: Main inference
        cudaStreamWaitEvent(main_stream, input_ready);
        launchKernel<<<grid, block, 0, main_stream>>>(kernelA);
        launchKernel<<<grid, block, 0, main_stream>>>(kernelB);
        
        // Stream 2: Async memory ops
        cudaMemcpyAsync(output, device, size, D2H, memory_stream);
        
        // Stream 3: Postprocessing
        launchKernel<<<grid, block, 0, postprocess_stream>>>(nms);
    }
};
```

**From our profile, we saw:**
- Stream 22: Main computation (~90 kernels)
- Stream 26: Additional operations
- Multiple streams for memory transfers

---

## Optimization Opportunities for RF-DETR

### 1. Batch Processing with Streams

```python
def optimized_batch_inference(model_id, images):
    """Process batch with stream concurrency."""
    from inference import get_model
    import torch
    
    model = get_model(model_id)
    
    # Create streams for each image
    streams = [torch.cuda.Stream() for _ in range(len(images))]
    results = []
    
    for img, stream in zip(images, streams):
        with torch.cuda.stream(stream):
            # Each image processes independently
            result = model.infer(img)
            results.append(result)
    
    # Synchronize all
    for stream in streams:
        stream.synchronize()
    
    return results

# Expected: 1.3-1.5x speedup for batch of 4
```

### 2. Pipeline Preprocessing

```python
def pipelined_inference(model, image_paths):
    """Pipeline: Load → Preprocess → Infer → Postprocess."""
    
    load_stream = torch.cuda.Stream()
    infer_stream = torch.cuda.Stream()
    post_stream = torch.cuda.Stream()
    
    for path in image_paths:
        with torch.cuda.stream(load_stream):
            img = load_and_preprocess(path)
        
        load_stream.synchronize()  # Wait for load
        
        with torch.cuda.stream(infer_stream):
            output = model.infer(img)
        
        infer_stream.synchronize()  # Wait for inference
        
        with torch.cuda.stream(post_stream):
            result = postprocess(output)
    
    return results
```

---

## Advanced: CUDA Graphs vs Streams

### CUDA Streams
- Submit operations dynamically
- Each kernel has launch overhead (~5-10μs)
- Flexible but has overhead

```python
for i in range(100):
    kernel()  # 100 launches × 10μs = 1ms overhead
```

### CUDA Graphs
- Capture entire sequence once
- Replay with minimal overhead (~1μs)
- Fast but requires static shapes

```python
# Capture
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for i in range(100):
        kernel()  # Captured

# Replay
graph.replay()  # 1μs overhead only!
```

**From our profile:**
- Current: 900 kernel launches in 10 iterations
- Each launch: ~252μs overhead
- Total overhead: 226.8ms (47.9% of API time)

**With CUDA graphs:** Could reduce overhead by 10-20%

---

## Stream Best Practices

### ✅ Do:

1. **Use pinned memory for async transfers**
```python
pinned = torch.cuda.pin_memory(tensor)
gpu_tensor = pinned.cuda(non_blocking=True)
```

2. **Create streams once, reuse**
```python
# Good: Create once
stream = torch.cuda.Stream()
for i in range(1000):
    with torch.cuda.stream(stream):
        process()

# Bad: Create every time
for i in range(1000):
    stream = torch.cuda.Stream()  # Wasteful!
    with torch.cuda.stream(stream):
        process()
```

3. **Synchronize only when necessary**
```python
# Good: Batch operations, then sync
for i in range(10):
    with torch.cuda.stream(stream):
        process(i)
stream.synchronize()  # Once at end

# Bad: Sync every iteration
for i in range(10):
    with torch.cuda.stream(stream):
        process(i)
    stream.synchronize()  # Defeats purpose!
```

### ❌ Don't:

1. **Don't create too many streams**
```python
# Bad: Too many streams (overhead)
streams = [torch.cuda.Stream() for _ in range(1000)]

# Good: Reasonable number
streams = [torch.cuda.Stream() for _ in range(4)]
```

2. **Don't forget to synchronize**
```python
# Bad: Race condition!
with torch.cuda.stream(stream):
    result = compute()
print(result)  # May not be ready!

# Good: Synchronize first
with torch.cuda.stream(stream):
    result = compute()
stream.synchronize()
print(result)  # Safe
```

---

## Debugging Streams

### Check Current Stream

```python
import torch

# Get current stream
current = torch.cuda.current_stream()
print(f"Stream: {current}")

# Check if stream is done
is_done = stream.query()  # True if all ops complete
```

### Profile with Nsight

```bash
# Profile with stream visualization
nsys profile --trace=cuda,nvtx,osrt python script.py

# View in nsys-ui - see different streams as rows
nsys-ui profile.nsys-rep
```

**What you'll see:**
- Each stream as a separate row
- Operations in each stream
- Overlapping execution across streams
- Gaps indicate synchronization points

---

## Summary

### What Are Streams?
- Independent queues of GPU operations
- Execute operations in order within stream
- Can overlap across different streams

### Why Use Them?
1. Hide memory transfer latency
2. Increase GPU utilization
3. Reduce CPU-GPU sync overhead
4. Enable concurrent processing

### Key Concepts
- **Sequential within stream:** Operations wait for previous
- **Concurrent across streams:** Different streams run in parallel
- **Synchronization needed:** CPU must wait when accessing results
- **Limited by hardware:** Can't exceed GPU resources

### In RF-DETR Context

**Current state:**
- TensorRT uses ~6 streams automatically
- Stream 22: Main inference kernels
- Other streams: Memory ops, postprocessing

**Optimization potential:**
- Batch processing with streams: 1.3-1.5x speedup
- Pipeline stages: Small improvement
- CUDA graphs: 1.1-1.2x speedup

---

## Further Reading

- [CUDA C Programming Guide - Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
- [PyTorch CUDA Streams Documentation](https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams)
- [TensorRT Best Practices - Streams](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimize-performance)

---

**Related to RF-DETR profiling:**
- See `NSIGHT_PROFILING_REPORT.md` for stream usage analysis
- See `timeline_376_380ms_analysis.md` for stream operations in postprocessing
- See `ADVANCED_TENSORRT_OPTIMIZATION_GUIDE.md` for stream-based optimizations
