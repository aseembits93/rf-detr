#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
Rebuild the rfdetr TRT engine locally on this GPU with maximum optimization.

Changes vs the prebuilt engine:
  1. Built on this exact GPU so TRT auto-tunes kernel tactics for the target arch.
  2. builder_optimization_level = 5 (max) — tries more tactics for each layer.
  3. All tactic sources enabled (cuBLAS, cuBLAS-LT, cuDNN, JIT convs, edge mask).
  4. Optional INT8 quantization with entropy calibration.
  5. Timing cache persistence for faster rebuilds.

Usage:
    # FP16 engine (default)
    python rebuild_trt_engine.py --onnx /tmp/cache/coco/40/weights.onnx

    # INT8+FP16 engine with calibration
    python rebuild_trt_engine.py --onnx /tmp/cache/coco/40/weights.onnx \
        --precision int8+fp16 --calibration-dir /path/to/calibration/npy/

    # Prepare calibration data first
    python rebuild_trt_engine.py --prepare-calibration \
        --images-dir /path/to/images/ --calibration-dir /tmp/calibration/
"""

import argparse
import os
import shutil
import sys
import time


def get_gpu_name() -> str:
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Rebuild TRT engine with optimized settings")
    parser.add_argument("--onnx", type=str, default="/tmp/cache/coco/40/weights.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--engine-dir", type=str, default=None,
                        help="Directory to save engine (default: same as ONNX)")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp32", "fp16", "int8", "int8+fp16"],
                        help="Engine precision mode")
    parser.add_argument("--optimization-level", type=int, default=5,
                        help="TRT builder optimization level (0-5)")
    parser.add_argument("--workspace-gb", type=int, default=8,
                        help="Max workspace memory in GB")
    parser.add_argument("--input-h", type=int, default=576)
    parser.add_argument("--input-w", type=int, default=576)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--calibration-dir", type=str, default=None,
                        help="Directory with .npy calibration files (for INT8)")
    parser.add_argument("--calibration-cache", type=str, default="calibration.cache",
                        help="Path for calibration cache file")
    parser.add_argument("--max-calibration-images", type=int, default=512)
    parser.add_argument("--timing-cache", type=str, default="timing.cache",
                        help="Path for timing cache (speeds up rebuilds)")

    parser.add_argument("--prepare-calibration", action="store_true",
                        help="Preprocess images for calibration instead of building engine")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Source image directory for --prepare-calibration")

    parser.add_argument("--backup", action="store_true", default=True,
                        help="Backup original engine before overwriting")

    args = parser.parse_args()

    if args.prepare_calibration:
        if not args.images_dir:
            parser.error("--images-dir required with --prepare-calibration")
        if not args.calibration_dir:
            parser.error("--calibration-dir required with --prepare-calibration")
        from rfdetr.export.tensorrt import prepare_calibration_data
        prepare_calibration_data(
            images_dir=args.images_dir,
            output_dir=args.calibration_dir,
            input_shape=(args.input_h, args.input_w),
            max_images=args.max_calibration_images,
        )
        return

    if "int8" in args.precision and not args.calibration_dir:
        print("WARNING: INT8 precision requested without --calibration-dir.", file=sys.stderr)
        print("Engine will be built without calibration data (accuracy may suffer).", file=sys.stderr)

    engine_dir = args.engine_dir or os.path.dirname(args.onnx)
    engine_name = os.path.basename(args.onnx).replace(".onnx", f"_{args.precision}.engine")
    engine_path = os.path.join(engine_dir, engine_name)

    if args.backup and os.path.exists(engine_path):
        backup_path = engine_path + ".bak"
        if not os.path.exists(backup_path):
            print(f"Backing up {engine_path} -> {backup_path}", file=sys.stderr)
            shutil.copy2(engine_path, backup_path)

    calibrator = None
    if "int8" in args.precision and args.calibration_dir:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — initializes CUDA context
        from rfdetr.export.tensorrt import create_int8_calibrator
        calibrator = create_int8_calibrator(
            calibration_dir=args.calibration_dir,
            cache_file=args.calibration_cache,
            batch_size=args.batch_size,
            max_images=args.max_calibration_images,
        )

    print(f"\nBuilding {args.precision.upper()} engine on {get_gpu_name()}", file=sys.stderr)
    print(f"  ONNX: {args.onnx}", file=sys.stderr)
    print(f"  Output: {engine_path}", file=sys.stderr)
    print(f"  Input shape: ({args.batch_size}, 3, {args.input_h}, {args.input_w})", file=sys.stderr)
    print(f"  Optimization level: {args.optimization_level}", file=sys.stderr)

    from rfdetr.export.tensorrt import build_engine
    build_engine(
        onnx_path=args.onnx,
        engine_path=engine_path,
        precision=args.precision,
        optimization_level=args.optimization_level,
        workspace_gb=args.workspace_gb,
        input_shape=(args.batch_size, 3, args.input_h, args.input_w),
        calibrator=calibrator,
        timing_cache_path=args.timing_cache,
    )

    print(f"\nDone. Engine saved to: {engine_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
