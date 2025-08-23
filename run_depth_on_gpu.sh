# My laptop runs the screen graphics by default on the Intel GPU,
# so I need to force the use of the NVIDIA GPU for the ZED Depth Viewer.
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia /usr/local/zed/tools/ZED_Depth_Viewer