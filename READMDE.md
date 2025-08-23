# ZED2 Camera Project

Cross-platform ZED2 stereo camera development setup for laptop development and Jetson Nano deployment.

## Project Structure

```
zed2_camera/
├── README.md
├── requirements-laptop.txt      # x86_64 laptop dependencies  
├── requirements-jetson.txt      # Jetson Nano dependencies
├── setup_laptop.sh             # Setup script for development
├── setup_jetson.sh             # Setup script for deployment
├── zed_view_on_laptop.py       # Main application
├── zed_view_no_opencv.py       # Fallback version (PIL-based)
└── wheels/
    └── pyzed-4.2-cp312-cp312-linux_x86_64.whl  # x86_64 wheel (for laptop)
```

## Quick Start

### For Laptop Development (x86_64)
```bash
./setup_laptop.sh
./run_zed2.sh
```

### For Jetson Nano Deployment (ARM64)  
```bash
# On Jetson Nano:
./setup_jetson.sh
~/run_zed2_jetson.sh
```

## Requirements

### Laptop (Development)
- Ubuntu 20.04+ or similar Linux distribution
- NVIDIA GPU with drivers installed
- ZED SDK 4.2 for x86_64
- Python 3.8+

### Jetson Nano (Deployment)
- JetPack 4.6+ (Ubuntu 18.04 based)
- ZED SDK 4.2 for L4T/ARM64
- Python 3.8+ (system version recommended)

## Manual Setup

### Laptop Environment
```bash
python3 -m venv .zed2_laptop_env
source .zed2_laptop_env/bin/activate
pip install -r requirements-laptop.txt
pip install wheels/pyzed-4.2-cp312-cp312-linux_x86_64.whl
```

### Jetson Environment  
```bash
# Use system Python (no venv needed)
pip3 install --user -r requirements-jetson.txt
cd /usr/local/zed && python3 get_python_api.py
```

## Running

### With GPU acceleration (laptop):
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python3 zed_view_on_laptop.py
```

### On Jetson Nano:
```bash
python3 zed_view_on_laptop.py
```

## Troubleshooting

### OpenCV Issues
- **Laptop**: Use OpenCV 4.8.1.78 (not 4.12.0+)
- **Jetson**: Prefer system OpenCV, fallback to pip version

### ZED SDK Installation
- **Laptop**: Download x86_64 version from StereoLabs
- **Jetson**: Download L4T/ARM64 version, not x86_64

### Performance
- **Laptop**: Requires NVIDIA GPU environment variables for optimal performance
- **Jetson**: GPU acceleration is automatic, no special environment variables needed

## Architecture Notes

The same Python code runs on both platforms, but:
- **Laptop**: Uses pip-installed dependencies in virtual environment
- **Jetson**: Uses system Python with user-installed packages for better stability
- **ZED Wheels**: Different architectures require different wheel files (x86_64 vs ARM64)