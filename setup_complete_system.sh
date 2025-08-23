#!/bin/bash
# Complete ZED2 Tracking System Setup Script
# Handles both laptop (x86_64) and Jetson Nano (ARM64) installations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PLATFORM=$(uname -m)

echo "=== ZED2 Intelligent Tracking System Setup ==="
echo "Platform: $PLATFORM"
echo "Directory: $SCRIPT_DIR"

# Detect platform
if [[ "$PLATFORM" == "x86_64" ]]; then
    SETUP_TYPE="laptop"
elif [[ "$PLATFORM" == "aarch64" ]]; then
    SETUP_TYPE="jetson"
else
    echo "Unsupported platform: $PLATFORM"
    exit 1
fi

echo "Setup type: $SETUP_TYPE"

# Check for ZED SDK
if [ ! -d "/usr/local/zed" ]; then
    echo "ERROR: ZED SDK not found at /usr/local/zed"
    echo ""
    if [[ "$SETUP_TYPE" == "laptop" ]]; then
        echo "Please install ZED SDK 4.2 for x86_64 from:"
        echo "https://www.stereolabs.com/developers/release/"
        echo "Download: ZED SDK 4.2 (Ubuntu 20/22)"
    else
        echo "Please install ZED SDK 4.2 for L4T/Jetson from:"
        echo "https://www.stereolabs.com/developers/release/"
        echo "Download: ZED SDK 4.2 (JetPack 4.6+)"
    fi
    exit 1
fi

echo "✓ ZED SDK found at /usr/local/zed"

# Setup based on platform
if [[ "$SETUP_TYPE" == "laptop" ]]; then
    echo ""
    echo "=== Setting up for Laptop Development ==="
    
    # Create virtual environment
    VENV_DIR=".zed2_complete_env"
    echo "Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    echo "Installing Python dependencies..."
    pip install numpy==1.26.4
    pip install opencv-python==4.8.1.78  # Specific working version
    pip install cython>=3.0.0
    pip install Pillow>=8.0.0
    
    # Install YOLO dependencies
    echo "Installing YOLO dependencies..."
    pip install ultralytics>=8.0.0
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    
    # Install point cloud processing
    echo "Installing point cloud processing..."
    pip install open3d>=0.15.0 matplotlib>=3.5.0 scipy>=1.7.0 scikit-learn>=1.0.0
    
    # Install ZED Python API
    echo "Installing ZED Python API..."
    if [ -f "wheels/pyzed-4.2-cp312-cp312-linux_x86_64.whl" ]; then
        echo "Using local ZED wheel file..."
        pip install wheels/pyzed-4.2-cp312-cp312-linux_x86_64.whl
    else
        echo "Downloading ZED Python API..."
        cd /usr/local/zed
        python3 get_python_api.py
        cd "$SCRIPT_DIR"
    fi
    
    # Create run script with GPU acceleration
    cat > run_tracking_laptop.sh << 'EOF'
#!/bin/bash
# Run complete tracking system on laptop with GPU acceleration
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Activate virtual environment
source .zed2_complete_env/bin/activate

# Run the system
python3 main_tracking_system.py "$@"
EOF
    
    chmod +x run_tracking_laptop.sh
    
    echo ""
    echo "✓ Laptop setup complete!"
    echo ""
    echo "Usage:"
    echo "  ./run_tracking_laptop.sh --generate-config  # Generate config"
    echo "  ./run_tracking_laptop.sh                     # Run tracking system"
    echo "  python3 zed_view_on_laptop.py               # Test basic ZED"
    
elif [[ "$SETUP_TYPE" == "jetson" ]]; then
    echo ""
    echo "=== Setting up for Jetson Nano Deployment ==="
    
    # Update system packages
    echo "Updating system packages..."
    sudo apt-get update
    
    # Install system dependencies
    echo "Installing system dependencies..."
    sudo apt-get install -y python3-pip python3-dev python3-numpy python3-opencv
    sudo apt-get install -y i2c-tools python3-smbus
    
    # Install Python dependencies (user installation for Jetson)
    echo "Installing Python dependencies..."
    pip3 install --user cython>=3.0.0
    pip3 install --user "numpy>=1.13,<2.0"
    pip3 install --user Pillow>=8.0.0
    
    # Try system OpenCV first, fallback to pip
    python3 -c "import cv2; print('System OpenCV version:', cv2.__version__)" || {
        echo "Installing OpenCV via pip..."
        pip3 install --user opencv-python==4.8.1.78
    }
    
    # Install YOLO for Jetson (lighter version)
    echo "Installing YOLO for Jetson..."
    pip3 install --user ultralytics
    pip3 install --user torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
    
    # Install point cloud processing (optional for Jetson)
    echo "Installing point cloud processing..."
    pip3 install --user matplotlib>=3.5.0 scipy>=1.7.0 scikit-learn>=1.0.0
    # Note: open3d might be too heavy for Jetson Nano, will use numpy fallback
    
    # Install servo control libraries
    echo "Installing servo control libraries..."
    pip3 install --user adafruit-circuitpython-pca9685
    pip3 install --user adafruit-blinka
    
    # Install ZED Python API for ARM64
    echo "Installing ZED Python API for Jetson..."
    cd /usr/local/zed
    python3 get_python_api.py
    cd "$SCRIPT_DIR"
    
    # Enable I2C
    echo "Enabling I2C..."
    sudo usermod -a -G i2c $USER
    
    # Create run script for Jetson
    cat > run_tracking_jetson.sh << 'EOF'
#!/bin/bash
# Run complete tracking system on Jetson Nano
export PYTHONPATH="${PYTHONPATH}:${HOME}/.local/lib/python3.8/site-packages"

# Run the system (no GPU offload variables needed on Jetson)
python3 main_tracking_system.py "$@"
EOF
    
    chmod +x run_tracking_jetson.sh
    
    echo ""
    echo "✓ Jetson Nano setup complete!"
    echo ""
    echo "Usage:"
    echo "  ./run_tracking_jetson.sh --generate-config  # Generate config"
    echo "  ./run_tracking_jetson.sh                     # Run tracking system"
    echo "  python3 pantilt_controller.py               # Test servos"
    echo ""
    echo "Hardware setup:"
    echo "  Connect PCA9685 to Jetson Nano I2C pins"
    echo "  Connect servos to PCA9685 channels 0 (pan) and 1 (tilt)"
    echo "  Reboot or re-login for I2C group membership"
fi

# Generate default configuration
echo "Generating default configuration..."
if [[ "$SETUP_TYPE" == "laptop" ]]; then
    source .zed2_complete_env/bin/activate
fi

python3 -c "
from main_tracking_system import IntegratedTrackingSystem
import json

# Create system and get default config
system = IntegratedTrackingSystem()
config = system.get_default_config()

# Adjust for platform
if '$SETUP_TYPE' == 'jetson':
    config['pantilt']['enabled'] = True
    config['pointcloud']['enabled'] = True  # But will use numpy fallback if needed
else:
    config['pantilt']['enabled'] = False  # No hardware on laptop
    config['pointcloud']['enabled'] = True

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Generated config.json')
"

# Create comprehensive test script
cat > test_system.sh << 'EOF'
#!/bin/bash
# Comprehensive system test script

echo "=== ZED2 System Test ==="

# Test ZED camera basic functionality
echo "1. Testing ZED camera..."
if [[ "$PLATFORM" == "x86_64" ]]; then
    __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python3 -c "
import pyzed.sl as sl
zed = sl.Camera()
init_params = sl.InitParameters()
err = zed.open(init_params)
if err == sl.ERROR_CODE.SUCCESS:
    print('✓ ZED camera OK')
    zed.close()
else:
    print('✗ ZED camera failed:', err)
"
else
    python3 -c "
import pyzed.sl as sl
zed = sl.Camera()
init_params = sl.InitParameters()
err = zed.open(init_params)
if err == sl.ERROR_CODE.SUCCESS:
    print('✓ ZED camera OK')
    zed.close()
else:
    print('✗ ZED camera failed:', err)
"
fi

# Test YOLO
echo "2. Testing YOLO..."
python3 -c "
try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    print('✓ YOLO OK')
except Exception as e:
    print('✗ YOLO failed:', e)
"

# Test point cloud processing
echo "3. Testing point cloud processing..."
python3 -c "
try:
    from pointcloud_processor import PointCloudProcessor
    processor = PointCloudProcessor()
    print('✓ Point cloud processing OK')
except Exception as e:
    print('✗ Point cloud processing failed:', e)
"

# Test servo control (Jetson only)
if [[ "$SETUP_TYPE" == "jetson" ]]; then
    echo "4. Testing servo control..."
    python3 -c "
try:
    from pantilt_controller import PanTiltController
    # Test initialization only (no actual movement)
    print('✓ Servo control libraries OK')
except Exception as e:
    print('✗ Servo control failed:', e)
"
fi

echo "Test complete!"
EOF

chmod +x test_system.sh

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Run system test: ./test_system.sh"
echo "2. Test basic ZED: python3 zed_view_on_laptop.py"
echo "3. Run full system: ./run_tracking_${SETUP_TYPE}.sh"
echo ""
echo "Configuration file: config.json"
echo "Edit the config to customize detection classes, hardware settings, etc."
echo ""
echo "For help: python3 main_tracking_system.py --help"