#!/usr/bin/env python3
"""Minimal script for ncu profiling of a single RF-DETR nano inference."""

import os
os.environ["ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND"] = "False"

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

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


def main():
    model = get_model("rfdetr-nano")

    response = requests.get("https://media.roboflow.com/inference/people-walking.jpg")
    image_np = np.array(Image.open(BytesIO(response.content)).convert("RGB"))

    # Warmup - ncu will skip these via --launch-skip
    for i in range(5):
        model.infer(image_np)
        print(f"Warmup {i+1}/5", file=sys.stderr)

    torch.cuda.profiler.start()
    print("Profiled inference starting...", file=sys.stderr)
    model.infer(image_np)
    torch.cuda.profiler.stop()
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
