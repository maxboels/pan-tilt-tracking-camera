#!/bin/bash
# Run complete tracking system on laptop with GPU acceleration
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Activate virtual environment
source .zed2_complete_env/bin/activate

# Run the system
python3 main_tracking_system.py "$@"
