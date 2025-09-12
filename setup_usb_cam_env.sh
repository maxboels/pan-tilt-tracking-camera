#!/bin/bash

# Script to set up virtual environment for pan-tilt-tracking-camera with USB camera

echo "Setting up virtual environment for pan-tilt-tracking-camera..."

# Remove old environment if it exists
if [ -d ".pan_tilt_env" ]; then
    echo "Removing existing .pan_tilt_env..."
    rm -rf .pan_tilt_env
fi

# Create new virtual environment
echo "Creating new virtual environment..."
python3 -m venv .pan_tilt_env

# Activate virtual environment
echo "Activating virtual environment..."
source .pan_tilt_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements_usb_cam.txt

echo "Virtual environment setup complete!"
echo "To activate the environment, run: source .pan_tilt_env/bin/activate"
echo "To deactivate, run: deactivate"