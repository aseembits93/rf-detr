#!/bin/bash
# ------------------------------------------------------------------------
# RF-DETR Profiling Setup Test Script
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

set -e

# Set up CUDA environment
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           RF-DETR Profiling Setup Test Suite                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check GPU
echo -e "${YELLOW}[1/7] Checking GPU availability...${NC}"
if nvidia-smi > /dev/null 2>&1; then
    echo -e "${GREEN}✓ GPU available${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "✗ GPU not available"
    exit 1
fi
echo

# Test 2: Check CUDA toolkit
echo -e "${YELLOW}[2/7] Checking CUDA toolkit...${NC}"
if [ -d "/usr/local/cuda-12.8" ]; then
    echo -e "${GREEN}✓ CUDA 12.8 installed${NC}"
    echo "Location: /usr/local/cuda-12.8"
else
    echo "✗ CUDA toolkit not found"
    exit 1
fi
echo

# Test 3: Check profiling tools
echo -e "${YELLOW}[3/7] Checking profiling tools...${NC}"
tools=(nsys ncu nvcc)
for tool in "${tools[@]}"; do
    if command -v $tool > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $tool${NC} available"
    else
        echo "✗ $tool not found"
        exit 1
    fi
done
echo

# Test 4: Check Python packages
echo -e "${YELLOW}[4/7] Checking Python packages...${NC}"
uv run --no-sync python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
uv run --no-sync python -c "import nsight; print(f'✓ nsight-python {nsight.__version__}')"
echo

# Test 5: Test profile_utils.py
echo -e "${YELLOW}[5/7] Testing profile_utils.py...${NC}"
uv run --no-sync python -c "
from profile_utils import nvtx_range, CUDATimer, profile_memory_usage
import torch
if torch.cuda.is_available():
    with nvtx_range('test'):
        x = torch.randn(100, 100, device='cuda')
    print('✓ profile_utils.py working')
else:
    print('✗ CUDA not available in Python')
    exit(1)
"
echo

# Test 6: Test profiling with profile_gpu.sh
echo -e "${YELLOW}[6/7] Testing profile_gpu.sh...${NC}"
./profile_gpu.sh -n test_suite -- uv run --no-sync python -c "
import torch
x = torch.randn(500, 500, device='cuda')
y = torch.matmul(x, x)
print('Profile test complete')
" > /dev/null 2>&1

if [ -f "profile_results/test_suite_nsys.nsys-rep" ]; then
    echo -e "${GREEN}✓ Profiling script working${NC}"
    echo "Profile saved: profile_results/test_suite_nsys.nsys-rep"
else
    echo "✗ Profile not created"
    exit 1
fi
echo

# Test 7: Test profile analysis
echo -e "${YELLOW}[7/7] Testing profile_nsight_analysis.py...${NC}"
uv run --no-sync python profile_nsight_analysis.py \
    profile_results/test_suite_nsys.nsys-rep \
    --summary-only > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Profile analysis working${NC}"
else
    echo "✗ Profile analysis failed"
    exit 1
fi
echo

# Summary
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                  ALL TESTS PASSED ✓                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo
echo "Profiling setup is fully functional!"
echo
echo "Available tools:"
echo "  • profile_gpu.sh          - Easy profiling wrapper"
echo "  • profile_utils.py        - Python profiling utilities"
echo "  • profile_nsight.py       - Nsight-python profiling"
echo "  • profile_nsight_analysis.py - Profile analysis"
echo
echo "Documentation:"
echo "  • PROFILING.md                - Main guide"
echo "  • PROFILING_QUICKREF.md       - Quick reference"
echo "  • PROFILING_NSIGHT_PYTHON.md  - Nsight Python guide"
echo
echo "Test profile created at: profile_results/test_suite_nsys.nsys-rep"
echo
echo "Try it:"
echo "  ./profile_gpu.sh -- python your_script.py"
echo "  python profile_nsight_analysis.py profile_results/*.nsys-rep"
