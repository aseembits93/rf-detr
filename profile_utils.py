# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GPU profiling utilities for RF-DETR."""

import contextlib
import time
from typing import Optional

import torch


@contextlib.contextmanager
def nvtx_range(msg: str, color: str = "blue"):
    """
    Context manager for NVTX ranges (visible in Nsight Systems).

    Args:
        msg: Label for this code region
        color: Color for the range in profiler (blue, green, yellow, red, etc.)
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(msg)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


class CUDATimer:
    """
    Accurate CUDA timing using CUDA events.

    Example:
        timer = CUDATimer()
        with timer:
            model(input)
        print(f"Time: {timer.elapsed_time_ms:.2f} ms")
    """

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self._elapsed_time_ms = None

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        self._elapsed_time_ms = self.start_event.elapsed_time(self.end_event)

    @property
    def elapsed_time_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._elapsed_time_ms is None:
            raise RuntimeError("Timer has not been used yet")
        return self._elapsed_time_ms


def benchmark_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> dict:
    """
    Benchmark model inference time.

    Args:
        model: PyTorch model
        input_tensor: Input tensor (should be on GPU)
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations

    Returns:
        Dictionary with timing statistics
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            timer = CUDATimer()
            with timer:
                _ = model(input_tensor)
            times.append(timer.elapsed_time_ms)

    return {
        "mean_ms": sum(times) / len(times),
        "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times))
        ** 0.5,
        "min_ms": min(times),
        "max_ms": max(times),
        "median_ms": sorted(times)[len(times) // 2],
        "num_iterations": num_iterations,
    }


def profile_memory_usage() -> dict:
    """
    Get current GPU memory usage statistics.

    Returns:
        Dictionary with memory statistics in MB
    """
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


def print_memory_usage(prefix: str = ""):
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    stats = profile_memory_usage()
    print(f"{prefix}GPU Memory:")
    print(f"  Allocated: {stats['allocated_mb']:.2f} MB")
    print(f"  Reserved:  {stats['reserved_mb']:.2f} MB")
    print(f"  Max Allocated: {stats['max_allocated_mb']:.2f} MB")
    print(f"  Max Reserved:  {stats['max_reserved_mb']:.2f} MB")


@contextlib.contextmanager
def profile_section(name: str, print_stats: bool = True):
    """
    Profile a code section with timing and memory tracking.

    Args:
        name: Name of the section
        print_stats: Whether to print statistics after the section

    Example:
        with profile_section("data_loading"):
            data = load_batch()
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = profile_memory_usage()

    start_time = time.time()
    torch.cuda.nvtx.range_push(name) if torch.cuda.is_available() else None

    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

        elapsed_time = time.time() - start_time

        if print_stats:
            print(f"\n[Profile] {name}")
            print(f"  Time: {elapsed_time * 1000:.2f} ms")

            if torch.cuda.is_available():
                end_mem = profile_memory_usage()
                print(f"  Memory allocated: {end_mem['allocated_mb']:.2f} MB")
                print(
                    f"  Peak memory: {end_mem['max_allocated_mb']:.2f} MB "
                    f"(+{end_mem['max_allocated_mb'] - start_mem['allocated_mb']:.2f} MB)"
                )


if __name__ == "__main__":
    # Example usage
    print("GPU Profiling Utilities for RF-DETR\n")

    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()

    # Example: Profile a simple operation
    print("Example: Profiling matrix multiplication")
    with profile_section("matmul_example"):
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        c = torch.matmul(a, b)

    print("\nMemory usage:")
    print_memory_usage()

    # Example: Benchmark operation
    print("\n" + "=" * 60)
    print("Example: Benchmarking matrix multiplication")

    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return torch.matmul(x, x.T)

    model = SimpleModel().cuda()
    input_tensor = torch.randn(1000, 1000, device="cuda")

    stats = benchmark_inference(model, input_tensor, num_warmup=10, num_iterations=100)

    print(f"Mean: {stats['mean_ms']:.2f} ms")
    print(f"Std:  {stats['std_ms']:.2f} ms")
    print(f"Min:  {stats['min_ms']:.2f} ms")
    print(f"Max:  {stats['max_ms']:.2f} ms")
