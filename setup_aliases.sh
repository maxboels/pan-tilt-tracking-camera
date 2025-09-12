#!/bin/bash

# Script to set up shell aliases for easy environment switching
# Run this script to add aliases to your shell configuration

SCRIPT_DIR="/home/maxboels/projects/pan-tilt-tracking-camera"
SHELL_NAME=$(basename "$SHELL")

# Determine which shell config file to use
case "$SHELL_NAME" in
    "bash")
        CONFIG_FILE="$HOME/.bashrc"
        ;;
    "zsh")
        CONFIG_FILE="$HOME/.zshrc"
        ;;
    *)
        echo "Unsupported shell: $SHELL_NAME"
        echo "Please manually add the aliases to your shell configuration file"
        exit 1
        ;;
esac

echo "Setting up aliases for $SHELL_NAME in $CONFIG_FILE..."

# Backup the original config file
cp "$CONFIG_FILE" "${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
echo "Backup created: ${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"

# Remove old pan-tilt aliases if they exist
sed -i '/# Pan-Tilt Tracking Camera aliases/,/# End Pan-Tilt aliases/d' "$CONFIG_FILE"

# Add new aliases
cat >> "$CONFIG_FILE" << 'EOF'

# Pan-Tilt Tracking Camera aliases
alias ptcam="cd /home/maxboels/projects/pan-tilt-tracking-camera && source select_env.sh --interactive"
alias ptcam-auto="cd /home/maxboels/projects/pan-tilt-tracking-camera && source select_env.sh --auto"
alias ptcam-usb="cd /home/maxboels/projects/pan-tilt-tracking-camera && source .pan_tilt_env/bin/activate"
alias ptcam-zed="cd /home/maxboels/projects/pan-tilt-tracking-camera && source .zed2_complete_env/bin/activate"
alias ptcam-reset="cd /home/maxboels/projects/pan-tilt-tracking-camera && source select_env.sh --reset"
# End Pan-Tilt aliases
EOF

echo ""
echo "Aliases added successfully!"
echo ""
echo "Available commands after restarting your terminal (or run 'source $CONFIG_FILE'):"
echo "  ptcam         - Interactive environment selection"
echo "  ptcam-auto    - Use saved default environment"
echo "  ptcam-usb     - Directly activate USB camera environment"
echo "  ptcam-zed     - Directly activate ZED2 camera environment"
echo "  ptcam-reset   - Reset environment preferences"
echo ""
echo "To apply changes now, run: source $CONFIG_FILE"