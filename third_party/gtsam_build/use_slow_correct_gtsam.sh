#!/usr/bin/env bash

# Source this file before running Python scripts that should use the locally
# rebuilt slow-but-correct GTSAM instead of the conda-installed package.

ROOT="/home/yuzhou/Desktop/abstraction-recovery/third_party/gtsam_build"
export PYTHONPATH="${ROOT}/build-slow/python${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${ROOT}/build-slow/gtsam${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Using slow-correct GTSAM from ${ROOT}/build-slow/python"
