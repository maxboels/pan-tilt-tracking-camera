#!/bin/bash

# Script to find and fix VSCode configurations for pan-tilt-tracking-camera
# This searches VSCode settings, workspace files, and terminal configurations

PROJECT_DIR="/home/maxboels/projects/pan-tilt-tracking-camera"
ZED_ACTIVATION="zed2_complete_env"

echo "🔍 Searching for ZED2 environment activation in VSCode configurations..."
echo ""

# VSCode configuration locations to check
VSCODE_LOCATIONS=(
    "$HOME/.config/Code/User/settings.json"
    "$HOME/.vscode/settings.json"
    "$PROJECT_DIR/.vscode/settings.json"
    "$PROJECT_DIR/.vscode/launch.json"
    "$PROJECT_DIR/.vscode/tasks.json"
    "$HOME/.config/Code/User/tasks.json"
    "$HOME/.config/Code/User/launch.json"
    "$HOME/.config/Code/CachedExtensions"
    "$HOME/.config/Code/logs"
)

# Workspace files
WORKSPACE_FILES=(
    "$PROJECT_DIR/pan-tilt-tracking-camera.code-workspace"
    "$PROJECT_DIR/zed2_camera.code-workspace"
    "$HOME/*.code-workspace"
)

found_files=()

echo "Checking VSCode settings files..."
for config_file in "${VSCODE_LOCATIONS[@]}"; do
    if [ -f "$config_file" ]; then
        if grep -q "$ZED_ACTIVATION" "$config_file" 2>/dev/null; then
            found_files+=("$config_file")
            echo "✓ Found ZED2 reference in: $config_file"
            echo "  Content:"
            grep -n "$ZED_ACTIVATION" "$config_file" 2>/dev/null | sed 's/^/    /'
            echo ""
        fi
    fi
done

echo "Checking workspace files..."
for workspace_file in "${WORKSPACE_FILES[@]}"; do
    if [ -f "$workspace_file" ]; then
        if grep -q "$ZED_ACTIVATION" "$workspace_file" 2>/dev/null; then
            found_files+=("$workspace_file")
            echo "✓ Found ZED2 reference in: $workspace_file"
            echo "  Content:"
            grep -n "$ZED_ACTIVATION" "$workspace_file" 2>/dev/null | sed 's/^/    /'
            echo ""
        fi
    fi
done

# Check for workspace files with glob expansion
for workspace_file in $HOME/*.code-workspace; do
    if [ -f "$workspace_file" ]; then
        if grep -q "$ZED_ACTIVATION" "$workspace_file" 2>/dev/null; then
            found_files+=("$workspace_file")
            echo "✓ Found ZED2 reference in: $workspace_file"
            echo "  Content:"
            grep -n "$ZED_ACTIVATION" "$workspace_file" 2>/dev/null | sed 's/^/    /'
            echo ""
        fi
    fi
done

# Check VSCode project settings
if [ -f "$PROJECT_DIR/.vscode/settings.json" ]; then
    echo "Current project VSCode settings:"
    echo "📁 $PROJECT_DIR/.vscode/settings.json"
    if [ -s "$PROJECT_DIR/.vscode/settings.json" ]; then
        cat "$PROJECT_DIR/.vscode/settings.json" | sed 's/^/    /'
    else
        echo "    (file is empty)"
    fi
    echo ""
fi

# Look for terminal configurations
echo "Checking for terminal auto-run configurations..."
if [ -f "$PROJECT_DIR/.vscode/settings.json" ]; then
    if grep -q "terminal\|python\|defaultProfile" "$PROJECT_DIR/.vscode/settings.json" 2>/dev/null; then
        echo "✓ Found terminal/Python configurations in project settings"
        grep -n -A2 -B2 "terminal\|python\|defaultProfile" "$PROJECT_DIR/.vscode/settings.json" 2>/dev/null | sed 's/^/    /'
        echo ""
    fi
fi

if [ ${#found_files[@]} -eq 0 ]; then
    echo "❌ No ZED2 references found in VSCode configuration files."
    echo ""
    echo "Let's run a broader search..."
    echo "🔍 Searching entire home directory (this may take a moment)..."
    
    # Broader search
    search_results=$(find "$HOME" -name "*.json" -o -name "*.code-workspace" 2>/dev/null | xargs grep -l "$ZED_ACTIVATION" 2>/dev/null)
    
    if [ -n "$search_results" ]; then
        echo "✓ Found ZED2 references in:"
        echo "$search_results" | sed 's/^/    /'
        echo ""
        
        echo "Showing content of found files:"
        echo "$search_results" | while read file; do
            echo "📁 $file:"
            grep -n "$ZED_ACTIVATION" "$file" 2>/dev/null | sed 's/^/    /'
            echo ""
        done
    else
        echo "❌ No ZED2 references found in JSON/workspace files"
        echo ""
        echo "The auto-activation might be coming from:"
        echo "  • VSCode extension settings"
        echo "  • Python extension auto-activation"
        echo "  • Terminal integration settings"
        echo "  • Conda/Mamba auto-activation"
        echo ""
        echo "Try these steps:"
        echo "  1. Check VSCode Python extension settings (Ctrl+,)"
        echo "  2. Look for 'Python: Default Interpreter Path'"
        echo "  3. Check 'Python: Terminal Activate Environment'"
        echo "  4. Look for any workspace-specific settings"
    fi
else
    echo "🎯 Found ${#found_files[@]} file(s) with ZED2 references!"
    echo ""
    
    read -p "Do you want to update these files? [y/N]: " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔧 Updating VSCode configurations..."
        
        for config_file in "${found_files[@]}"; do
            # Create backup
            backup_file="${config_file}.backup.$(date +%Y%m%d_%H%M%S)"
            cp "$config_file" "$backup_file"
            echo "📋 Backup created: $backup_file"
            
            # Replace zed2_complete_env with pan_tilt_env
            sed -i 's/\.zed2_complete_env/\.pan_tilt_env/g' "$config_file"
            sed -i 's/zed2_complete_env/pan_tilt_env/g' "$config_file"
            echo "✅ Updated: $config_file"
        done
        
        echo ""
        echo "🎉 VSCode configurations updated!"
        echo "   ZED2 references replaced with USB camera environment"
        echo ""
        echo "Next steps:"
        echo "  1. Restart VSCode"
        echo "  2. Check that the new environment activates correctly"
        echo "  3. If issues persist, check Python extension settings"
    else
        echo "No changes made."
    fi
fi

# Create/update project VSCode settings for new environment
echo ""
echo "📝 Creating recommended VSCode settings for the project..."

# Ensure .vscode directory exists
mkdir -p "$PROJECT_DIR/.vscode"

# Create or update settings.json
cat > "$PROJECT_DIR/.vscode/settings.json" << 'EOF'
{
    "python.defaultInterpreterPath": "./.pan_tilt_env/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.terminal.activateEnvInCurrentTerminal": true,
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${workspaceFolder}"
    },
    "files.associations": {
        "*.py": "python"
    },
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.analysis.typeCheckingMode": "basic"
}
EOF

echo "✅ Created/updated .vscode/settings.json with USB camera environment settings"

# Create launch configuration
cat > "$PROJECT_DIR/.vscode/launch.json" << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "USB Camera Test",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/usb_camera.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "python": "${workspaceFolder}/.pan_tilt_env/bin/python"
        },
        {
            "name": "Pan-Tilt Tracking",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main_usb_tracking.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "python": "${workspaceFolder}/.pan_tilt_env/bin/python",
            "args": ["--camera", "0", "--fps", "30"]
        }
    ]
}
EOF

echo "✅ Created/updated .vscode/launch.json with debug configurations"

echo ""
echo "🎯 VSCode setup complete! The project is now configured to use the USB camera environment."