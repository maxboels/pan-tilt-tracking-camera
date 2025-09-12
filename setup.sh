#!/bin/bash
# Setup script for Pan-Tilt Tracking Camera

set -e

echo "Setting up Pan-Tilt Tracking Camera environment..."

# Create virtual environment
ENV_NAME=".pan_tilt_env"
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment: $ENV_NAME"
    python3 -m venv $ENV_NAME
else
    echo "Virtual environment already exists: $ENV_NAME"
fi

# Activate environment
source $ENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Download YOLO model if not present
if [ ! -f "yolov8n.pt" ]; then
    echo "Downloading YOLOv8 nano model..."
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
fi

# Create directories
mkdir -p captures
mkdir -p logs

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source $ENV_NAME/bin/activate"
echo ""
echo "To run the tracking system:"
echo "  python main.py"
echo ""
echo "To test individual components:"
echo "  python src/yolo_tracker.py  # Test YOLO tracking"
echo "  python src/usb_camera.py    # Test camera"
echo "
