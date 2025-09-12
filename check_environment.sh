#!/bin/bash

# Script to check and optimize environment activation
# This will help clean up any duplicate activations and conda conflicts

PROJECT_DIR="/home/maxboels/projects/pan-tilt-tracking-camera"

echo "🔍 Checking current environment status..."
echo ""

# Check what's currently active
echo "Current environment status:"
echo "  VIRTUAL_ENV: ${VIRTUAL_ENV:-'Not set'}"
echo "  CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-'Not set'}"
echo "  PATH: ${PATH:0:100}..." 
echo "  PYTHONPATH: ${PYTHONPATH:-'Not set'}"
echo ""

# Check if conda is initialized
if command -v conda &> /dev/null; then
    echo "✓ Conda is available"
    echo "  Conda version: $(conda --version)"
    echo "  Active environment: ${CONDA_DEFAULT_ENV:-'base'}"
    echo ""
    
    echo "Conda configuration:"
    conda config --show | grep -E "(auto_activate_base|changeps1)" 2>/dev/null || echo "  No relevant conda config found"
    echo ""
else
    echo "❌ Conda not found"
    echo ""
fi

# Check current Python and pip
echo "Current Python setup:"
if command -v python &> /dev/null; then
    echo "  Python path: $(which python)"
    echo "  Python version: $(python --version)"
else
    echo "  ❌ Python not found"
fi

if command -v pip &> /dev/null; then
    echo "  Pip path: $(which pip)"
else
    echo "  ❌ Pip not found"
fi
echo ""

# Check for conflicting activations
echo "🔍 Checking for environment conflicts..."

# Count how many environments are active
env_count=0
active_envs=()

if [ -n "$VIRTUAL_ENV" ]; then
    env_count=$((env_count + 1))
    active_envs+=("Virtual Environment: $VIRTUAL_ENV")
fi

if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    env_count=$((env_count + 1))
    active_envs+=("Conda Environment: $CONDA_DEFAULT_ENV")
fi

if [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    active_envs+=("Conda Base Environment")
fi

echo "Active environments detected: $env_count"
for env in "${active_envs[@]}"; do
    echo "  • $env"
done
echo ""

if [ $env_count -gt 1 ]; then
    echo "⚠️  WARNING: Multiple environments are active!"
    echo "   This can cause package conflicts and unexpected behavior."
    echo ""
    
    echo "Recommended actions:"
    echo "1. Deactivate conda base auto-activation"
    echo "2. Use only the pan_tilt_env for this project"
    echo ""
    
    read -p "Do you want to fix conda auto-activation? [y/N]: " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔧 Configuring conda..."
        
        # Disable conda base auto-activation
        conda config --set auto_activate_base false
        echo "✅ Disabled conda base auto-activation"
        
        # Set changeps1 to false to clean up prompt
        conda config --set changeps1 false
        echo "✅ Disabled conda prompt changes"
        
        echo ""
        echo "✅ Conda configuration updated!"
        echo ""
        echo "Changes made:"
        echo "  • Conda base environment will not auto-activate"
        echo "  • Conda will not modify your shell prompt"
        echo ""
        echo "To apply changes:"
        echo "  1. Close and reopen your terminal"
        echo "  2. The pan_tilt_env should activate cleanly"
        echo ""
        echo "If you need conda later, you can manually activate it with:"
        echo "  conda activate base"
    fi
elif [ $env_count -eq 1 ] && [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Perfect! Only the virtual environment is active."
    echo "   Environment: $(basename "$VIRTUAL_ENV")"
    
    # Check if it's the right environment
    if [[ "$VIRTUAL_ENV" == *".pan_tilt_env"* ]]; then
        echo "✅ Correct environment is active!"
    else
        echo "⚠️  Different virtual environment is active"
        echo "   Expected: .pan_tilt_env"
        echo "   Actual: $(basename "$VIRTUAL_ENV")"
    fi
elif [ $env_count -eq 0 ]; then
    echo "❌ No virtual environment is active"
    echo "   This is unusual if VSCode is auto-activating"
fi

echo ""
echo "🔧 Environment optimization recommendations:"

# Check if we're in the project directory
if [ "$PWD" = "$PROJECT_DIR" ]; then
    echo "✅ You're in the correct project directory"
else
    echo "⚠️  You're not in the project directory"
    echo "   Current: $PWD"
    echo "   Expected: $PROJECT_DIR"
fi

# Check if the pan_tilt_env exists and is properly set up
if [ -d "$PROJECT_DIR/.pan_tilt_env" ]; then
    echo "✅ Pan-tilt environment exists"
    
    # Check if it has the required packages
    if [ -f "$PROJECT_DIR/.pan_tilt_env/bin/pip" ]; then
        echo "✅ Environment has pip installed"
        
        # Check for key packages
        echo "📦 Checking installed packages..."
        "$PROJECT_DIR/.pan_tilt_env/bin/pip" list | grep -E "(opencv|mediapipe|numpy)" || echo "   ⚠️  Some packages may be missing"
    else
        echo "❌ Environment setup incomplete"
    fi
else
    echo "❌ Pan-tilt environment not found"
    echo "   Run: ./setup_usb_cam_env.sh"
fi

echo ""
echo "🎯 Summary and next steps:"

if [ $env_count -gt 1 ]; then
    echo "1. Restart your terminal after conda configuration changes"
    echo "2. Verify only pan_tilt_env is active"
    echo "3. Test the camera: python usb_camera.py"
elif [[ "$VIRTUAL_ENV" == *".pan_tilt_env"* ]]; then
    echo "✅ Environment setup looks good!"
    echo "You can now run:"
    echo "  python usb_camera.py          # Test camera"
    echo "  python main_usb_tracking.py   # Run tracking"
else
    echo "1. Ensure you're in the project directory"
    echo "2. Activate the environment: source .pan_tilt_env/bin/activate"
    echo "3. Test the setup"
fi