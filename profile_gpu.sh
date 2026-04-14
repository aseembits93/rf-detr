#!/bin/bash
# ------------------------------------------------------------------------
# RF-DETR GPU Profiling Helper Script
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
PROFILER="nsys"
OUTPUT_DIR="./profile_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Help message
show_help() {
    cat << EOF
GPU Profiling Helper for RF-DETR

Usage: $0 [OPTIONS] -- <python command>

OPTIONS:
    -p, --profiler TYPE     Profiler to use: nsys, ncu, or both (default: nsys)
    -o, --output DIR        Output directory (default: ./profile_results)
    -n, --name NAME         Custom name for output files (default: profile_<timestamp>)
    -h, --help              Show this help message

PROFILER TYPES:
    nsys    - Nsight Systems: Timeline profiling, good for overall performance
    ncu     - Nsight Compute: Kernel-level profiling, detailed metrics
    both    - Run both profilers sequentially

EXAMPLES:
    # Profile training with Nsight Systems
    $0 -- python -m rfdetr.train --config configs/rtdetr_r50vd_6x.yml

    # Profile inference with Nsight Compute
    $0 -p ncu -- python benchmark_inference.py

    # Profile with custom output name
    $0 -n my_experiment -- python train.py

    # Run both profilers
    $0 -p both -- python benchmark.py

ENVIRONMENT VARIABLES:
    NSYS_OPTS   - Additional options for nsys (e.g., NSYS_OPTS="--trace=cuda,nvtx")
    NCU_OPTS    - Additional options for ncu (e.g., NCU_OPTS="--set full")

EOF
}

# Parse arguments
NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--profiler)
            PROFILER="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--name)
            NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if command was provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No command provided${NC}"
    show_help
    exit 1
fi

# Set output name
if [ -z "$NAME" ]; then
    NAME="profile_${TIMESTAMP}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo -e "${RED}Error: nvidia-smi not found. Is NVIDIA driver installed?${NC}"
    exit 1
fi

echo -e "${GREEN}=== GPU Profiling Setup ===${NC}"
echo "Profiler: $PROFILER"
echo "Output Directory: $OUTPUT_DIR"
echo "Output Name: $NAME"
echo "Command: $@"
echo

# Run profiling based on selected profiler
run_nsys() {
    local output_file="${OUTPUT_DIR}/${NAME}_nsys"
    echo -e "${YELLOW}Running Nsight Systems profiling...${NC}"

    # Default nsys options
    local nsys_opts="--trace=cuda,nvtx,cudnn,cublas --cudabacktrace=true --python-sampling=true"

    # Add user-specified options
    if [ -n "$NSYS_OPTS" ]; then
        nsys_opts="$nsys_opts $NSYS_OPTS"
    fi

    nsys profile $nsys_opts --output="$output_file" "$@"

    echo -e "${GREEN}✓ Nsight Systems profiling complete${NC}"
    echo "Output: ${output_file}.nsys-rep"
    echo
    echo "To view the results:"
    echo "  nsys-ui ${output_file}.nsys-rep  # GUI (if X11 forwarding enabled)"
    echo "  nsys stats ${output_file}.nsys-rep  # Command-line statistics"
    echo
}

run_ncu() {
    local output_file="${OUTPUT_DIR}/${NAME}_ncu"
    echo -e "${YELLOW}Running Nsight Compute profiling...${NC}"

    # Default ncu options (lightweight set for quick profiling)
    local ncu_opts="--set basic"

    # Add user-specified options
    if [ -n "$NCU_OPTS" ]; then
        ncu_opts="$NCU_OPTS"
    fi

    echo -e "${YELLOW}Note: Nsight Compute can be slow. Using 'basic' metric set.${NC}"
    echo -e "${YELLOW}For detailed analysis, run with: NCU_OPTS=\"--set full\"${NC}"
    echo

    ncu $ncu_opts --export="$output_file" "$@"

    echo -e "${GREEN}✓ Nsight Compute profiling complete${NC}"
    echo "Output: ${output_file}.ncu-rep"
    echo
    echo "To view the results:"
    echo "  ncu-ui ${output_file}.ncu-rep  # GUI (if X11 forwarding enabled)"
    echo "  ncu --import ${output_file}.ncu-rep --page raw  # Command-line statistics"
    echo
}

# Execute based on profiler choice
case $PROFILER in
    nsys)
        run_nsys "$@"
        ;;
    ncu)
        run_ncu "$@"
        ;;
    both)
        run_nsys "$@"
        echo -e "${YELLOW}---${NC}"
        run_ncu "$@"
        ;;
    *)
        echo -e "${RED}Error: Invalid profiler '$PROFILER'. Choose: nsys, ncu, or both${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}=== Profiling Complete ===${NC}"
echo "All results saved to: $OUTPUT_DIR"
