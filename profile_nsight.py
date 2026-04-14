#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
GPU profiling script using NVIDIA Nsight Python SDK.

This script provides programmatic profiling capabilities using the nsight-python
package, which offers a high-level Python API for NVTX annotations and profiling
workflow automation.
"""

import argparse
import contextlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch

try:
    from nsight import annotation
except ImportError:
    print("Error: nsight-python package not found.")
    print("Install with: uv sync --group profiling")
    sys.exit(1)


class NsightProfiler:
    """
    High-level profiler using Nsight Python SDK.

    Provides automatic NVTX annotations and profiling context management.
    """

    def __init__(self, output_dir: str = "./profile_results", name: str = "profile"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.profile_path = self.output_dir / f"{name}.nsys-rep"

    @contextlib.contextmanager
    def profile(
        self,
        trace_options: Optional[list] = None,
        auto_start: bool = True,
    ):
        """
        Context manager for profiling with Nsight Systems.

        Args:
            trace_options: Additional trace options (e.g., ["cuda", "nvtx", "cudnn"])
            auto_start: Whether to automatically start profiling

        Example:
            with profiler.profile():
                model(input)
        """
        if trace_options is None:
            trace_options = ["cuda", "nvtx", "cudnn", "cublas"]

        trace_str = ",".join(trace_options)

        # Build nsys command
        cmd = [
            "nsys",
            "profile",
            f"--trace={trace_str}",
            "--cudabacktrace=true",
            "--python-sampling=true",
            f"--output={self.profile_path.with_suffix('')}",
            "--force-overwrite=true",
        ]

        if not auto_start:
            cmd.append("--capture-range=cudaProfilerApi")

        print(f"Starting Nsight Systems profiling...")
        print(f"Output: {self.profile_path}")
        print(f"Trace: {trace_str}")
        print()

        # Start profiling
        if torch.cuda.is_available():
            torch.cuda.cudart().cudaProfilerStart()

        yield

        # Stop profiling
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

    @staticmethod
    @contextlib.contextmanager
    def annotate(name: str, color: Optional[str] = None):
        """
        Annotate a code region with NVTX markers via nsight-python.

        Args:
            name: Region name
            color: Optional color for visualization

        Example:
            with NsightProfiler.annotate("forward_pass", "green"):
                output = model(input)
        """
        with annotation.annotate(name):
            yield

    def generate_stats(self):
        """Generate statistics report from profile."""
        if not self.profile_path.exists():
            print(f"Warning: Profile not found at {self.profile_path}")
            return

        print("\n" + "=" * 70)
        print("Generating profile statistics...")
        print("=" * 70)

        try:
            # CUDA API summary
            print("\n--- CUDA API Summary ---")
            subprocess.run(
                ["nsys", "stats", str(self.profile_path), "--report", "cuda_api_sum"],
                check=True,
            )

            # CUDA GPU kernel summary
            print("\n--- CUDA GPU Kernel Summary ---")
            subprocess.run(
                [
                    "nsys",
                    "stats",
                    str(self.profile_path),
                    "--report",
                    "cuda_gpu_kern_sum",
                ],
                check=True,
            )

        except subprocess.CalledProcessError as e:
            print(f"Error generating stats: {e}")


def benchmark_with_profiling(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 100,
    profiler: Optional[NsightProfiler] = None,
) -> dict:
    """
    Benchmark model with automatic NVTX annotations.

    Args:
        model: PyTorch model
        input_tensor: Input tensor
        num_warmup: Warmup iterations
        num_iterations: Benchmark iterations
        profiler: Optional NsightProfiler instance

    Returns:
        Timing statistics
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    model.eval()

    # Warmup (using torch.cuda.nvtx since nsight doesn't support nested annotations)
    torch.cuda.nvtx.range_push("warmup")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    # Benchmark
    times = []
    torch.cuda.nvtx.range_push("benchmark")
    for i in range(num_iterations):
        torch.cuda.nvtx.range_push(f"iteration_{i}")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            _ = model(input_tensor)
        end_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()

    return {
        "mean_ms": sum(times) / len(times),
        "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times))
        ** 0.5,
        "min_ms": min(times),
        "max_ms": max(times),
        "median_ms": sorted(times)[len(times) // 2],
    }


def profile_training_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: dict,
    optimizer: torch.optim.Optimizer,
):
    """
    Profile a single training step with detailed annotations.

    Args:
        model: PyTorch model
        inputs: Input tensor
        targets: Target dict
        optimizer: Optimizer
    """
    # Use torch.cuda.nvtx for nested annotations
    torch.cuda.nvtx.range_push("training_step")

    # Forward pass
    torch.cuda.nvtx.range_push("forward")
    outputs = model(inputs, targets)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs
    torch.cuda.nvtx.range_pop()

    # Backward pass
    torch.cuda.nvtx.range_push("backward")
    loss.backward()
    torch.cuda.nvtx.range_pop()

    # Optimizer step
    torch.cuda.nvtx.range_push("optimizer_step")
    optimizer.step()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("optimizer_zero_grad")
    optimizer.zero_grad()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    return loss.item() if torch.is_tensor(loss) else loss


def main():
    parser = argparse.ArgumentParser(
        description="Profile PyTorch models with Nsight Python SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile inference benchmark
  python profile_nsight.py --mode benchmark --model-type simple

  # Profile training step
  python profile_nsight.py --mode training

  # Custom profiling with specific traces
  python profile_nsight.py --mode benchmark --trace cuda,nvtx --iterations 50

  # Generate stats from existing profile
  python profile_nsight.py --stats-only --profile profile_results/my_profile.nsys-rep
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["benchmark", "training", "custom"],
        default="benchmark",
        help="Profiling mode",
    )
    parser.add_argument(
        "--model-type",
        choices=["simple", "rfdetr"],
        default="simple",
        help="Model type to profile",
    )
    parser.add_argument("--output-dir", default="./profile_results", help="Output directory")
    parser.add_argument("--name", default="nsight_profile", help="Profile name")
    parser.add_argument(
        "--trace",
        default="cuda,nvtx,cudnn,cublas",
        help="Trace options (comma-separated)",
    )
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only generate stats from existing profile",
    )
    parser.add_argument("--profile", help="Path to existing profile for stats-only mode")

    args = parser.parse_args()

    # Stats-only mode
    if args.stats_only:
        if not args.profile:
            print("Error: --profile required for --stats-only mode")
            sys.exit(1)
        profiler = NsightProfiler(output_dir=Path(args.profile).parent, name="")
        profiler.profile_path = Path(args.profile)
        profiler.generate_stats()
        return

    # Check CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()

    # Initialize profiler
    profiler = NsightProfiler(output_dir=args.output_dir, name=args.name)

    # Create model
    if args.model_type == "simple":
        print("Creating simple model (matrix multiplication)...")

        class SimpleModel(torch.nn.Module):
            def __init__(self, size=1000):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(size, size))

            def forward(self, x):
                return torch.matmul(x, self.weight)

        model = SimpleModel().cuda()
        input_tensor = torch.randn(args.batch_size, 1000, device="cuda")

    elif args.model_type == "rfdetr":
        print("Creating RF-DETR model...")
        try:
            from rfdetr import RTDETR

            model = RTDETR("rtdetr-r18vd").cuda().eval()
            input_tensor = torch.randn(args.batch_size, 3, 640, 640, device="cuda")
        except ImportError:
            print("Error: RF-DETR not available. Use --model-type simple")
            sys.exit(1)
    else:
        print(f"Error: Unknown model type {args.model_type}")
        sys.exit(1)

    # Run profiling based on mode
    if args.mode == "benchmark":
        print(f"\nRunning benchmark mode ({args.iterations} iterations)...")
        print("=" * 70)

        # Run benchmark with annotations
        stats = benchmark_with_profiling(
            model,
            input_tensor,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
            profiler=profiler,
        )

        print("\nBenchmark Results:")
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Std:  {stats['std_ms']:.2f} ms")
        print(f"  Min:  {stats['min_ms']:.2f} ms")
        print(f"  Max:  {stats['max_ms']:.2f} ms")

    elif args.mode == "training":
        print("\nRunning training mode (single step)...")
        print("=" * 70)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Dummy targets
        targets = {"dummy": torch.randn(args.batch_size, device="cuda")}

        # Profile training step
        loss = profile_training_step(model, input_tensor, targets, optimizer)
        print(f"\nTraining step complete. Loss: {loss:.4f}")

    print(f"\nProfile saved to: {profiler.profile_path}")
    print("\nTo view the profile:")
    print(f"  nsys-ui {profiler.profile_path}")
    print(f"  nsys stats {profiler.profile_path}")


if __name__ == "__main__":
    main()
