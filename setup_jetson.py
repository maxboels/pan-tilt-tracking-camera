#!/bin/bash
# ZED2 Setup for Jetson Nano
# Run this on the Jetson Nano

set -e

echo "Setting up ZED2 Camera Environment for Jetson Nano..."

# Check if we're on ARM64 (Jetson)
if [[ $(uname -m) != "aarch64" ]]; then
    echo "WARNING: This script is designed for Jetson Nano (ARM64)"
    echo "Current architecture: $(uname -m)"
fi

# Check if ZED SDK is installed
if [ ! -d "/usr/local/zed" ]; then
    echo "ERROR: ZED SDK not found at /usr/local/zed"
    echo "Please install ZED SDK for Jetson from:"
    echo "https://www.stereolabs.com/developers/release/"
    echo "Download the L4T (Jetson) version, not x86_64"
    exit 1
fi

# Use system Python (recommended for Jetson)
echo "Using system Python for Jetson compatibility"

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-numpy python3-opencv

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user cython>=3.0.0
pip3 install --user "numpy>=1.13,<2.0"  # Force compatible version
pip3 install --user Pillow>=8.0.0

# Try to use system OpenCV first, fallback to pip if needed
python3 -c "import cv2; print('System OpenCV version:', cv2.__version__)" || {
    echo "System OpenCV not working, installing via pip..."
    pip3 install --user opencv-python==4.8.1.78
}

# Install ZED Python API for ARM64
echo "Installing ZED Python API for Jetson..."
cd /usr/local/zed
python3 get_python_api.py

# Create run script
echo "Creating run script..."
cat > ~/run_zed2_jetson.sh << 'EOF'
#!/bin/bash
# Run ZED2 application on Jetson Nano
export PYTHONPATH="${PYTHONPATH}:${HOME}/.local/lib/python3.8/site-packages"

# Navigate to project directory
cd ~/zed2_camera

# Run the application (no GPU offload variables needed on Jetson)
python3 zed_view_on_laptop.py "$@"
EOF

chmod +x ~/run_zed2_jetson.sh

echo ""
echo "Jetson Nano setup complete!"
echo ""
echo "To use:"
echo "1. Clone your project: git clone <your-repo> ~/zed2_camera"
echo "2. Connect ZED2 camera" 
echo "3. Run: ~/run_zed2_jetson.sh"