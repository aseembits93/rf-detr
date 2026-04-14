#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
Profile analysis using NVIDIA Nsight Python SDK.

This script demonstrates automated profiling workflow with the nsight-python
package for analyzing existing Nsight Systems profiles.
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from nsight import analyze, extraction

    NSIGHT_AVAILABLE = True
except ImportError:
    NSIGHT_AVAILABLE = False
    print("Error: nsight-python package not found.")
    print("Install with: uv sync --group profiling")
    sys.exit(1)


def export_profile_to_sqlite(nsys_rep_path: Path) -> Path:
    """
    Export .nsys-rep to SQLite format for analysis.

    Args:
        nsys_rep_path: Path to .nsys-rep file

    Returns:
        Path to generated SQLite file
    """
    sqlite_path = nsys_rep_path.with_suffix(".sqlite")

    if sqlite_path.exists():
        print(f"SQLite file already exists: {sqlite_path}")
        return sqlite_path

    print(f"Exporting {nsys_rep_path} to SQLite...")
    try:
        subprocess.run(
            ["nsys", "export", "--type=sqlite", str(nsys_rep_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✓ Created: {sqlite_path}")
        return sqlite_path
    except subprocess.CalledProcessError as e:
        print(f"Error exporting to SQLite: {e}")
        print(e.stderr)
        sys.exit(1)


def analyze_cuda_api_calls(sqlite_path: Path):
    """
    Analyze CUDA API calls from profile.

    Args:
        sqlite_path: Path to SQLite profile database
    """
    print("\n" + "=" * 70)
    print("CUDA API Call Analysis")
    print("=" * 70)

    try:
        # Use nsight extraction API
        result = extraction.get_cuda_api_calls(str(sqlite_path))

        if PANDAS_AVAILABLE and hasattr(result, "to_pandas"):
            df = result.to_pandas()
            print(f"\nTotal CUDA API calls: {len(df)}")

            # Top 10 most frequent calls
            print("\nTop 10 most frequent CUDA API calls:")
            print(df["nameId"].value_counts().head(10))

            # Top 10 slowest calls
            if "duration" in df.columns:
                print("\nTop 10 slowest CUDA API calls:")
                slowest = df.nlargest(10, "duration")[["nameId", "duration"]]
                print(slowest)
        else:
            print(f"Found {len(result)} CUDA API calls")

    except Exception as e:
        print(f"Error analyzing CUDA API calls: {e}")
        print("Note: This feature may require specific nsight-python version/configuration")


def analyze_cuda_kernels(sqlite_path: Path):
    """
    Analyze CUDA kernel execution from profile.

    Args:
        sqlite_path: Path to SQLite profile database
    """
    print("\n" + "=" * 70)
    print("CUDA Kernel Analysis")
    print("=" * 70)

    try:
        # Use nsight extraction API
        result = extraction.get_cuda_kernels(str(sqlite_path))

        if PANDAS_AVAILABLE and hasattr(result, "to_pandas"):
            df = result.to_pandas()
            print(f"\nTotal kernel launches: {len(df)}")

            if "duration" in df.columns:
                total_kernel_time = df["duration"].sum()
                print(f"Total kernel time: {total_kernel_time / 1e6:.2f} ms")

                # Top kernels by time
                print("\nTop 10 kernels by execution time:")
                if "shortName" in df.columns:
                    kernel_times = df.groupby("shortName")["duration"].sum().sort_values(
                        ascending=False
                    )
                    print(kernel_times.head(10))
                else:
                    print("Kernel names not available in this profile")
        else:
            print(f"Found {len(result)} kernel launches")

    except Exception as e:
        print(f"Error analyzing kernels: {e}")
        print("Note: This feature may require specific nsight-python version/configuration")


def analyze_memory_operations(sqlite_path: Path):
    """
    Analyze memory transfer operations.

    Args:
        sqlite_path: Path to SQLite profile database
    """
    print("\n" + "=" * 70)
    print("Memory Transfer Analysis")
    print("=" * 70)

    try:
        # Use nsight extraction API
        result = extraction.get_memory_operations(str(sqlite_path))

        if PANDAS_AVAILABLE and hasattr(result, "to_pandas"):
            df = result.to_pandas()
            print(f"\nTotal memory operations: {len(df)}")

            if "bytes" in df.columns:
                total_bytes = df["bytes"].sum()
                print(f"Total data transferred: {total_bytes / 1024**2:.2f} MB")

                if "copyKind" in df.columns:
                    print("\nTransfers by type:")
                    print(df.groupby("copyKind")["bytes"].sum() / 1024**2)
        else:
            print(f"Found {len(result)} memory operations")

    except Exception as e:
        print(f"Error analyzing memory operations: {e}")
        print("Note: This feature may require specific nsight-python version/configuration")


def generate_summary_report(nsys_rep_path: Path):
    """
    Generate comprehensive summary report.

    Args:
        nsys_rep_path: Path to .nsys-rep file
    """
    print("\n" + "=" * 70)
    print("Summary Report")
    print("=" * 70)

    # Use nsys stats for summary
    reports = [
        ("CUDA API Summary", "cuda_api_sum"),
        ("CUDA GPU Kernel Summary", "cuda_gpu_kern_sum"),
        ("CUDA Memory Operations", "cuda_gpu_mem_size_sum"),
        ("CUDA Memory Time", "cuda_gpu_mem_time_sum"),
    ]

    for title, report_name in reports:
        print(f"\n--- {title} ---")
        try:
            result = subprocess.run(
                ["nsys", "stats", str(nsys_rep_path), "--report", report_name],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                # Print first 30 lines
                lines = result.stdout.split("\n")[:30]
                print("\n".join(lines))
            else:
                print(f"Report not available: {report_name}")
        except Exception as e:
            print(f"Error generating {title}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Nsight Systems profiles using nsight-python SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze existing profile
  python profile_nsight_analysis.py profile_results/my_profile.nsys-rep

  # Generate summary report only
  python profile_nsight_analysis.py profile.nsys-rep --summary-only

  # Full analysis with all components
  python profile_nsight_analysis.py profile.nsys-rep --full
        """,
    )

    parser.add_argument("profile", type=Path, help="Path to .nsys-rep profile file")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate summary report only (using nsys stats)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full analysis including extraction APIs (may be slow)",
    )
    parser.add_argument(
        "--export-sqlite",
        action="store_true",
        help="Export to SQLite and exit",
    )

    args = parser.parse_args()

    # Check profile exists
    if not args.profile.exists():
        print(f"Error: Profile not found: {args.profile}")
        sys.exit(1)

    print(f"Analyzing profile: {args.profile}")
    print(f"File size: {args.profile.stat().st_size / 1024**2:.2f} MB")

    # Export to SQLite
    sqlite_path = export_profile_to_sqlite(args.profile)

    if args.export_sqlite:
        print(f"\n✓ SQLite export complete: {sqlite_path}")
        return

    # Generate summary report
    if args.summary_only or not args.full:
        generate_summary_report(args.profile)

    # Full analysis using nsight-python extraction APIs
    if args.full:
        print("\nRunning detailed analysis using nsight-python extraction APIs...")
        print("(This may take a while for large profiles)")

        if not PANDAS_AVAILABLE:
            print("\nWarning: pandas not available. Install for better analysis:")
            print("  uv add pandas --group profiling")

        analyze_cuda_api_calls(sqlite_path)
        analyze_cuda_kernels(sqlite_path)
        analyze_memory_operations(sqlite_path)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    print(f"\nProfile: {args.profile}")
    print(f"SQLite DB: {sqlite_path}")
    print("\nTo view interactively:")
    print(f"  nsys-ui {args.profile}")


if __name__ == "__main__":
    main()
