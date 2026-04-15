# Advanced TensorRT Optimization Guide for RF-DETR Medium

## Current Performance Baseline

**Tesla T4 Performance:**
- Current: **12.58ms** average
- Min: 10.18ms
- P95: 14.23ms

**Goal:** Push to **3-6ms** range through advanced optimizations

---

## Optimization Roadmap

### Quick Wins (No Engine Rebuild)

| Optimization | Expected Speedup | Difficulty | Time to Implement |
|--------------|------------------|------------|-------------------|
| **Batch Inference** | **1.6x (8ms per image)** | Easy | 1 hour |
| GPU Preprocessing | 1.5x (~6ms savings) | Medium | 1-2 days |
| CUDA Graphs | 1.2x | Medium | 2-3 days |

### Advanced (Requires Engine Rebuild)

| Optimization | Expected Speedup | Difficulty | Time to Implement |
|--------------|------------------|------------|-------------------|
| **INT8 Quantization** | **2.5-3x (~4-5ms)** | Hard | 1-2 weeks |
| Dynamic Shapes | 1.15x | Easy | 1-2 days |
| Kernel Tuning | 1.1-1.2x | Very Hard | 2-4 weeks |

### Ultimate Performance

**Combined optimizations:** 3-4x speedup
- Expected: **3-6ms** on Tesla T4
- With A100: **1-2ms**
- With H100: **0.5-1ms**

---

## 1. Batch Inference (Easiest - 1.6x Speedup)

### Current Results

```
Batch 1: 12.58ms per image
Batch 2:  8.10ms per image (1.55x speedup) ✓
Batch 4:  8.01ms per image (1.57x speedup) ✓
Batch 8:  8.19ms per image (1.54x speedup) ✓
```

### Implementation

**Option A: Process Multiple Images**
```python
from inference import get_model
import numpy as np

model = get_model("rfdetr-medium")
images = [image1, image2, image3, image4]

# Sequential (slow)
results = [model.infer(img) for img in images]
# Total: 12.58ms × 4 = 50.32ms

# Batched (fast - if application supports it)
# Theoretical: 8.01ms × 4 = 32.04ms
# Speedup: 1.57x
```

**Option B: Async Processing with Thread Pool**
```python
from concurrent.futures import ThreadPoolExecutor
from inference import get_model

model = get_model("rfdetr-medium")

def process_image(image):
    return model.infer(image)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, images))
```

**When to Use:**
- Processing video streams
- Batch API endpoints
- Offline dataset processing

**Limitations:**
- Requires multiple images available simultaneously
- May increase per-request latency in single-image scenarios

---

## 2. INT8 Quantization (Biggest Speedup - 2.5-3x)

### Expected Performance

```
Current (FP16): 12.58ms
With INT8: ~4-5ms (2.5-3x speedup)
Accuracy loss: -0.5% to -2% mAP (typically)
```

### Implementation Steps

#### Step 1: Export to ONNX

```python
from rfdetr.variants import RFDETRMedium
import torch

# Load model
model = RFDETRMedium()

# Export to ONNX
model.export(
    "rfdetr_medium.onnx",
    simplify=True,
    opset_version=17
)

print("✓ ONNX model exported")
```

#### Step 2: Create Calibration Dataset

```python
import os
import numpy as np
from PIL import Image
from glob import glob

# Collect 500-1000 representative images
calibration_images = []
image_paths = glob("/path/to/calibration/images/*.jpg")[:1000]

for img_path in image_paths:
    img = Image.open(img_path).convert('RGB')
    img = img.resize((1920, 1080))  # Match your inference size
    img_array = np.array(img)
    calibration_images.append(img_array)

# Save calibration data
np.save("calibration_data.npy", calibration_images)
print(f"✓ Saved {len(calibration_images)} calibration images")
```

#### Step 3: Create Calibration Script

```python
# calibrate_int8.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class RFDETRCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data_path, cache_file, batch_size=1):
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.calibration_data = np.load(calibration_data_path)
        self.batch_size = batch_size
        self.current_index = 0
        self.cache_file = cache_file
        
        # Allocate GPU memory for batch
        self.device_input = cuda.mem_alloc(
            self.calibration_data[0].nbytes * batch_size
        )
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.calibration_data):
            return None
        
        batch = self.calibration_data[
            self.current_index:self.current_index + self.batch_size
        ]
        
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Usage
calibrator = RFDETRCalibrator(
    "calibration_data.npy",
    "rfdetr_medium_int8.cache",
    batch_size=1
)
```

#### Step 4: Build INT8 TensorRT Engine

**Option A: Using trtexec (Simplest)**
```bash
trtexec \
    --onnx=rfdetr_medium.onnx \
    --int8 \
    --saveEngine=rfdetr_medium_int8.engine \
    --verbose \
    --workspace=4096  # 4GB workspace
```

**Option B: Using Python API (More Control)**
```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_int8_engine(onnx_file, engine_file, calibrator):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_file, 'rb') as model:
        parser.parse(model.read())
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    # Enable INT8
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator
    
    # Build engine
    engine = builder.build_serialized_network(network, config)
    
    # Save engine
    with open(engine_file, 'wb') as f:
        f.write(engine)
    
    print(f"✓ INT8 engine saved to {engine_file}")

# Build
calibrator = RFDETRCalibrator(
    "calibration_data.npy",
    "rfdetr_medium_int8.cache"
)

build_int8_engine(
    "rfdetr_medium.onnx",
    "rfdetr_medium_int8.engine",
    calibrator
)
```

#### Step 5: Run Inference with INT8 Engine

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTInference:
    def __init__(self, engine_path):
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def infer(self, image):
        # Preprocess image
        input_data = preprocess(image)  # Your preprocessing
        
        # Copy to GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        # Run inference
        self.context.execute_v2(bindings=self.bindings)
        
        # Copy results back
        for output in self.outputs:
            cuda.memcpy_dtoh(output['host'], output['device'])
        
        return self.outputs[0]['host']

# Usage
trt_model = TRTInference("rfdetr_medium_int8.engine")
result = trt_model.infer(image)
```

### Validation

```python
# Compare FP16 vs INT8 accuracy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Run both models on validation set
fp16_results = run_inference(model_fp16, val_images)
int8_results = run_inference(model_int8, val_images)

# Evaluate
coco_gt = COCO("val_annotations.json")

for name, results in [("FP16", fp16_results), ("INT8", int8_results)]:
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    print(f"\n{name} mAP@0.5:0.95 = {coco_eval.stats[0]:.4f}")
```

**Expected Results:**
```
FP16 mAP@0.5:0.95 = 0.XXX
INT8 mAP@0.5:0.95 = 0.XXX (typically -0.5% to -2%)
```

---

## 3. GPU-Side Preprocessing (1.5x Speedup)

### Current Bottleneck

From Nsight profiling:
- 88.7% of memory transfer time is Host-to-Device
- ~6ms spent on CPU preprocessing + H2D transfer

### Implementation

```python
import torch
import torch.nn.functional as F

class GPUPreprocessor:
    def __init__(self, device='cuda'):
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    def preprocess(self, image_path_or_bytes):
        """Preprocess entirely on GPU."""
        # Option 1: Use torchvision.io (GPU decoder)
        import torchvision.io as io
        image = io.read_image(image_path_or_bytes).to(self.device)
        
        # Option 2: Use NVIDIA DALI (fastest)
        # See DALI section below
        
        # Resize on GPU
        image = F.interpolate(
            image.unsqueeze(0).float() / 255.0,
            size=(1080, 1920),
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize on GPU
        image = (image - self.mean) / self.std
        
        return image

# Usage
preprocessor = GPUPreprocessor()

# Measure speedup
import time
t0 = time.perf_counter()
tensor = preprocessor.preprocess("image.jpg")
# Run inference with tensor
t1 = time.perf_counter()
print(f"GPU preprocessing: {(t1-t0)*1000:.2f}ms")
```

### Using NVIDIA DALI (Maximum Speed)

```python
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def create_dali_pipeline(image_dir, batch_size):
    images, labels = fn.readers.file(
        file_root=image_dir,
        random_shuffle=False,
        name="Reader"
    )
    
    # Decode on GPU
    images = fn.decoders.image(
        images,
        device="mixed",  # GPU decoder
        output_type=types.RGB
    )
    
    # Resize on GPU
    images = fn.resize(
        images,
        resize_x=1920,
        resize_y=1080,
        interp_type=types.INTERP_LINEAR
    )
    
    # Normalize on GPU
    images = fn.normalize(
        images,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        stddev=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )
    
    return images

# Build pipeline
pipe = create_dali_pipeline(
    image_dir="/path/to/images",
    batch_size=4,
    device_id=0,
    num_threads=4
)
pipe.build()

# Get batch (all preprocessing on GPU)
outputs = pipe.run()
images = outputs[0].as_cpu()  # or .as_tensor() for GPU
```

**Expected Speedup:**
```
Before: 12.58ms (with CPU preprocessing)
After: ~6-7ms (with GPU preprocessing)
Speedup: 1.8-2x
```

---

## 4. CUDA Graphs (1.2x Speedup)

### What are CUDA Graphs?

CUDA Graphs capture the entire sequence of CUDA operations (kernels, memory copies) and replay them with minimal CPU overhead.

**Benefits:**
- Reduce kernel launch overhead (~10-20% speedup)
- Lower CPU usage
- More consistent latency

### Implementation

```python
import torch

class CUDAGraphInference:
    def __init__(self, trt_model):
        self.model = trt_model
        self.graph = None
        self.static_input = None
        self.static_output = None
        
    def warmup_and_capture(self, example_input):
        """Capture CUDA graph after warmup."""
        # Warmup (required before capture)
        for _ in range(10):
            _ = self.model.infer(example_input)
        
        torch.cuda.synchronize()
        
        # Allocate static tensors
        self.static_input = torch.from_numpy(example_input).cuda()
        
        # Begin capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            # Run inference (will be captured)
            self.static_output = self.model.infer(
                self.static_input.cpu().numpy()
            )
        
        torch.cuda.synchronize()
        print("✓ CUDA graph captured")
    
    def infer(self, image):
        """Run inference using captured graph."""
        # Copy input to static buffer
        np.copyto(self.static_input.cpu().numpy(), image)
        
        # Replay graph (very fast!)
        self.graph.replay()
        
        return self.static_output

# Usage
model = get_model("rfdetr-medium")
cuda_graph_model = CUDAGraphInference(model)

# Capture graph once
cuda_graph_model.warmup_and_capture(example_image)

# Run inference (1.2x faster!)
result = cuda_graph_model.infer(test_image)
```

**Limitations:**
- Requires fixed input shapes
- Cannot use with dynamic control flow
- Must use static memory allocations

**Expected Performance:**
```
Without CUDA Graphs: 12.58ms
With CUDA Graphs: ~10.5ms
Speedup: 1.2x
```

---

## 5. Combined Optimization Pipeline

### Ultimate Performance Setup

```python
import torch
import tensorrt as trt
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn

class OptimizedRFDETR:
    """
    Combines all optimizations:
    - INT8 quantization
    - GPU preprocessing (DALI)
    - CUDA Graphs
    - Batch inference
    """
    
    def __init__(self, engine_path, batch_size=4):
        self.batch_size = batch_size
        
        # Load INT8 engine
        self.trt_model = TRTInference(engine_path)
        
        # Setup DALI preprocessing
        self.dali_pipe = self.create_preprocessing_pipeline()
        
        # Capture CUDA graph
        self.cuda_graph = self.capture_cuda_graph()
    
    def create_preprocessing_pipeline(self):
        @pipeline_def
        def preprocess_pipeline():
            # DALI pipeline for GPU preprocessing
            images = fn.external_source(name="images")
            images = fn.resize(images, resize_x=1920, resize_y=1080)
            images = fn.normalize(images, mean=[...], stddev=[...])
            return images
        
        pipe = preprocess_pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            device_id=0
        )
        pipe.build()
        return pipe
    
    def capture_cuda_graph(self):
        # Capture CUDA graph for inference
        graph = torch.cuda.CUDAGraph()
        # ... capture logic ...
        return graph
    
    def infer_batch(self, images):
        """Process batch of images with all optimizations."""
        # Preprocess on GPU (DALI)
        preprocessed = self.dali_pipe.run()
        
        # Run INT8 inference with CUDA graphs
        results = self.cuda_graph.replay()
        
        return results

# Usage
optimized_model = OptimizedRFDETR(
    "rfdetr_medium_int8.engine",
    batch_size=4
)

results = optimized_model.infer_batch(images)
```

**Expected Performance:**
```
Baseline (FP16, single):     12.58ms per image
+ INT8:                       ~4.5ms per image (2.8x)
+ GPU preprocessing:          ~3.8ms per image (3.3x)
+ CUDA graphs:                ~3.2ms per image (3.9x)
+ Batch=4:                    ~2.5ms per image (5.0x)

Final: ~2.5-3ms per image on Tesla T4
```

---

## 6. Hardware Upgrade Options

### Performance Comparison

| Hardware | Expected Latency | Speedup vs T4 | Cost | Power |
|----------|------------------|---------------|------|-------|
| Tesla T4 (current) | 12.58ms | 1.0x | $2,000 | 70W |
| Tesla T4 (optimized) | 3-4ms | 3-4x | - | 70W |
| NVIDIA L4 | 8-10ms | 1.3x | $3,000 | 72W |
| NVIDIA A10 | 6-8ms | 1.6-2x | $4,000 | 150W |
| NVIDIA A100 40GB | 3-4ms | 3-4x | $10,000 | 250W |
| NVIDIA A100 80GB | 2.5-3.5ms | 3.6-5x | $15,000 | 300W |
| NVIDIA H100 80GB | 1.5-2ms | 6-8x | $30,000 | 350W |

### Recommendations

**Best Value:**
- Optimize current T4 with INT8 + batching → 3-4ms
- Cost: $0 (just engineering time)

**Best Performance/Watt:**
- NVIDIA L4 (new generation, similar power to T4)
- 8-10ms with FP16, 3-4ms with INT8

**Best Raw Performance:**
- NVIDIA H100 → 1.5-2ms
- But 15x more expensive than T4

---

## Implementation Checklist

### Phase 1: Quick Wins (1 week)
- [ ] Implement batch inference for video processing
- [ ] Measure baseline with Nsight
- [ ] Profile memory transfers

### Phase 2: GPU Preprocessing (1-2 weeks)
- [ ] Set up NVIDIA DALI
- [ ] Implement GPU preprocessing pipeline
- [ ] Validate accuracy
- [ ] Measure speedup

### Phase 3: INT8 Quantization (2-3 weeks)
- [ ] Export model to ONNX
- [ ] Create calibration dataset (500-1000 images)
- [ ] Build INT8 TensorRT engine
- [ ] Validate accuracy (ensure <2% mAP loss)
- [ ] Integrate INT8 engine into inference pipeline
- [ ] Benchmark performance

### Phase 4: CUDA Graphs (1 week)
- [ ] Implement CUDA graph capture
- [ ] Test with static shapes
- [ ] Measure launch overhead reduction

### Phase 5: Integration (1 week)
- [ ] Combine all optimizations
- [ ] End-to-end testing
- [ ] Performance validation
- [ ] Production deployment

**Total Timeline:** 6-8 weeks for full optimization pipeline

---

## Expected Results Summary

| Configuration | Latency | Speedup | Implementation |
|---------------|---------|---------|----------------|
| **Baseline (FP16)** | **12.58ms** | **1.0x** | Current |
| + Batch=4 | 8.01ms | 1.57x | Easy (1 day) |
| + INT8 | 4-5ms | 2.5-3x | Hard (2-3 weeks) |
| + GPU Preproc | 3-4ms | 3-4x | Medium (1-2 weeks) |
| + CUDA Graphs | 3-3.5ms | 3.6-4.3x | Medium (1 week) |
| **All Combined** | **2.5-3ms** | **4-5x** | 6-8 weeks |
| **+ A100 Hardware** | **1-1.5ms** | **8-12x** | + Hardware cost |
| **+ H100 Hardware** | **0.5-1ms** | **12-25x** | + Hardware cost |

---

## Troubleshooting

### INT8 Accuracy Loss Too High

```python
# Use mixed precision (INT8 + FP16)
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)

# Mark sensitive layers to stay in FP16
for layer in network:
    if layer.name in sensitive_layers:
        layer.precision = trt.float16
```

### CUDA Graph Capture Fails

```bash
# Ensure static shapes
export CUDA_LAUNCH_BLOCKING=1

# Check for dynamic control flow
# CUDA graphs require deterministic execution
```

### DALI Installation Issues

```bash
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
```

---

## Conclusion

**Recommended Path:**

1. **Start with batching** (if applicable) → 1.6x speedup (1 day)
2. **Implement INT8** → 2.5-3x total speedup (2-3 weeks)
3. **Add GPU preprocessing** if still needed → 3-4x total speedup
4. **Consider hardware upgrade** only if software optimizations insufficient

**Realistic Target on Tesla T4:**
- With INT8 + batching: **3-4ms** per image
- Full optimization: **2.5-3ms** per image
- **4-5x faster than current 12.58ms**

This is near the theoretical limit for T4 hardware. Further improvements require newer GPUs (A100/H100).
