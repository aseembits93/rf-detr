#!/usr/bin/env python3
"""
Profile rfdetr-medium inference using NVIDIA Nsight Systems.

This script runs inference multiple times to capture:
- GPU kernel execution
- CUDA API calls
- Memory transfers
- TensorRT engine execution

Usage:
    nsys profile -o rfdetr_medium_profile python profile_rfdetr_medium.py
"""

import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

# Ensure we import inference from the local ../inference checkout, not site-packages
_expected_inference_root = Path(__file__).resolve().parent.parent / "inference"
import inference

_actual_inference_root = Path(inference.__file__).resolve().parent.parent
if _actual_inference_root != _expected_inference_root:
    print(
        f"WARNING: inference imported from {_actual_inference_root}, "
        f"expected {_expected_inference_root}",
        file=sys.stderr,
    )
    sys.exit(1)

from inference import get_model


def download_test_image(url: str = "https://media.roboflow.com/inference/people-walking.jpg") -> np.ndarray:
    """Download and prepare test image."""
    print("Downloading test image...", file=sys.stderr)
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image_np = np.array(image)
    print(f"Image ready: {image_np.shape}", file=sys.stderr)
    return image_np


def main():
    print("=" * 80, file=sys.stderr)
    print("RF-DETR Medium Profiling", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # Load model
    print("\nLoading model...", file=sys.stderr)
    model = get_model("rfdetr-medium")

    # Check model type
    if hasattr(model, '_model'):
        print(f"Model type: {type(model._model).__name__}", file=sys.stderr)

    # Download image
    image_np = download_test_image()

    # Warmup (not profiled)
    print("\nWarming up (5 iterations)...", file=sys.stderr)
    for i in range(5):
        model.infer(image_np)
        print(f"  Warmup {i+1}/5", file=sys.stderr)

    # # Synchronize before profiling
    # torch.cuda.synchronize()
    print("\nStarting profiled inference runs...", file=sys.stderr)

    # ==== PROFILING REGION START ====
    # Use NVTX markers for better visualization
    torch.cuda.nvtx.range_push("rfdetr_medium_inference_batch")

    # Run 10 iterations for profiling
    num_iterations = 20
    times = []

    for i in range(num_iterations):
        torch.cuda.nvtx.range_push(f"inference_iteration_{i+1}")

        #torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Actual inference call
        results = model.infer(image_np)

        #torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        times.append(elapsed_ms)

        torch.cuda.nvtx.range_pop()  # End iteration marker

        print(f"  Iteration {i+1}/{num_iterations}: {elapsed_ms:.2f}ms", file=sys.stderr)

    torch.cuda.nvtx.range_pop()  # End batch marker
    # ==== PROFILING REGION END ====

    # Print summary
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\n" + "=" * 80, file=sys.stderr)
    print("PROFILING SUMMARY", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Average latency: {avg_time:.2f}ms", file=sys.stderr)
    print(f"Min latency:     {min_time:.2f}ms", file=sys.stderr)
    print(f"Max latency:     {max_time:.2f}ms", file=sys.stderr)
    print(f"Iterations:      {num_iterations}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)


if __name__ == "__main__":
    main()
