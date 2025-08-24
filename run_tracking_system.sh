#!/bin/bash
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Use the POSIX-compliant '.' command instead of 'source'
. .zed2_complete_env/bin/activate

python3 complete_tracking_with_servos.py