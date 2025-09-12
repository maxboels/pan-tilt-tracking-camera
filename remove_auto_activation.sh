#!/bin/bash

# Script to remove automatic ZED2 environment activation
# This will help find and remove the auto-activation from your shell config

PROJECT_DIR="/home/maxboels/projects/pan-tilt-tracking-camera"
ZED_ACTIVATION="source /home/maxboels/projects/pan-tilt-tracking-camera/.zed2_complete_env/bin/activate"

echo "Searching for automatic ZED2 environment activation..."
echo "Looking for: $ZED_ACTIVATION"
echo ""

# Common shell configuration files to check
CONFIG_FILES=(
    "$HOME/.bashrc"
    "$HOME/.zshrc"
    "$HOME/.profile"
    "$HOME/.bash_profile"
    "$HOME/.zprofile"
)

# VSCode configuration files to check
VSCODE_FILES=(
    "$HOME/.config/Code/User/settings.json"
    "$HOME/.vscode/settings.json"
    "$PROJECT_DIR/.vscode/settings.json"
    "$PROJECT_DIR/.vscode/launch.json"
    "$PROJECT_DIR/.vscode/tasks.json"
    "$PROJECT_DIR/pan-tilt-tracking-camera.code-workspace"
    "$PROJECT_DIR/zed2_camera.code-workspace"
)

found_files=()

# Search each config file
echo "Checking shell configuration files..."
for config_file in "${CONFIG_FILES[@]}"; do
    if [ -f "$config_file" ]; then
        if grep -q "zed2_complete_env" "$config_file"; then
            found_files+=("$config_file")
            echo "✓ Found ZED2 activation in: $config_file"
            
            # Show the lines containing the activation
            echo "  Lines found:"
            grep -n "zed2_complete_env" "$config_file" | sed 's/^/    /'
            echo ""
        fi
    fi
done

# Search VSCode files
echo "Checking VSCode configuration files..."
for config_file in "${VSCODE_FILES[@]}"; do
    if [ -f "$config_file" ]; then
        if grep -q "zed2_complete_env" "$config_file" 2>/dev/null; then
            found_files+=("$config_file")
            echo "✓ Found ZED2 activation in: $config_file"
            
            # Show the lines containing the activation
            echo "  Lines found:"
            grep -n "zed2_complete_env" "$config_file" | sed 's/^/    /'
            echo ""
        fi
    fi
done

if [ ${#found_files[@]} -eq 0 ]; then
    echo "No automatic ZED2 activation found in shell or VSCode config files."
    echo ""
    echo "Try running the VSCode-specific search script:"
    echo "  chmod +x fix_vscode_config.sh"
    echo "  ./fix_vscode_config.sh"
    echo ""
    echo "Or search manually with:"
    echo "  grep -r 'zed2_complete_env' ~/ 2>/dev/null"
    echo "  find ~/.config/Code -name '*.json' -exec grep -l 'zed2_complete_env' {} \\;"
    exit 0
fi

echo "Found automatic ZED2 activation in ${#found_files[@]} file(s)."
echo ""

# Ask user what to do
read -p "Do you want to remove the automatic activation? [y/N]: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing automatic ZED2 activation..."
    
    for config_file in "${found_files[@]}"; do
        # Create backup
        backup_file="${config_file}.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$config_file" "$backup_file"
        echo "Backup created: $backup_file"
        
        # Remove lines containing zed2_complete_env
        sed -i '/zed2_complete_env/d' "$config_file"
        echo "Removed ZED2 activation from: $config_file"
    done
    
    echo ""
    echo "✓ Automatic ZED2 activation removed!"
    echo ""
    echo "Next steps:"
    echo "1. Close and reopen your terminal"
    echo "2. Run the setup script: chmod +x setup_aliases.sh && ./setup_aliases.sh"
    echo "3. Use 'ptcam' to choose your environment interactively"
    
else
    echo "No changes made. The automatic activation is still in place."
    echo ""
    echo "If you want to manually edit the files, they are located at:"
    for config_file in "${found_files[@]}"; do
        echo "  $config_file"
    done
fi

echo ""
echo "After removing the automatic activation, you can:"
echo "- Use 'ptcam' for interactive environment selection"
echo "- Use 'ptcam-auto' to use your saved default"
echo "- Use 'ptcam-usb' to directly activate the USB camera environment"