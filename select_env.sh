#!/bin/bash

# Environment Selection Script for Pan-Tilt Tracking Camera
# This script allows you to choose which virtual environment to activate

PROJECT_DIR="/home/maxboels/projects/pan-tilt-tracking-camera"
CONFIG_FILE="$PROJECT_DIR/.env_choice"

# ANSI Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to display the header
show_header() {
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}             Pan-Tilt Tracking Camera Environment Setup           ${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
}

# Function to check if environment exists
check_env_exists() {
    local env_path="$1"
    if [ -d "$env_path" ]; then
        return 0
    else
        return 1
    fi
}

# Function to activate environment
activate_env() {
    local env_path="$1"
    local env_name="$2"
    
    if check_env_exists "$env_path"; then
        echo -e "${GREEN}Activating $env_name environment...${NC}"
        source "$env_path/bin/activate"
        echo -e "${GREEN}✓ Environment activated successfully!${NC}"
        echo -e "${BLUE}You are now using: $env_name${NC}"
        return 0
    else
        echo -e "${RED}✗ Environment not found: $env_path${NC}"
        return 1
    fi
}

# Function to show environment status
show_env_status() {
    echo -e "\n${YELLOW}Available Environments:${NC}"
    
    if check_env_exists "$PROJECT_DIR/.pan_tilt_env"; then
        echo -e "${GREEN}✓${NC} USB Camera Environment (.pan_tilt_env) - ${BLUE}Recommended${NC}"
    else
        echo -e "${RED}✗${NC} USB Camera Environment (.pan_tilt_env) - Not found"
    fi
    
    if check_env_exists "$PROJECT_DIR/.zed2_complete_env"; then
        echo -e "${GREEN}✓${NC} ZED2 Camera Environment (.zed2_complete_env) - ${YELLOW}Legacy${NC}"
    else
        echo -e "${RED}✗${NC} ZED2 Camera Environment (.zed2_complete_env) - Not found"
    fi
    
    echo ""
}

# Function to save user choice
save_choice() {
    echo "$1" > "$CONFIG_FILE"
}

# Function to load saved choice
load_choice() {
    if [ -f "$CONFIG_FILE" ]; then
        cat "$CONFIG_FILE"
    else
        echo "usb"  # default
    fi
}

# Function for interactive mode
interactive_mode() {
    show_header
    show_env_status
    
    echo -e "${PURPLE}Choose your environment:${NC}"
    echo -e "  ${GREEN}1)${NC} USB Camera Environment (.pan_tilt_env) ${BLUE}[Recommended]${NC}"
    echo -e "  ${GREEN}2)${NC} ZED2 Camera Environment (.zed2_complete_env) ${YELLOW}[Legacy]${NC}"
    echo -e "  ${GREEN}3)${NC} No environment (stay in base)"
    echo -e "  ${GREEN}4)${NC} Set as default and activate"
    echo -e "  ${GREEN}q)${NC} Quit without activating"
    echo ""
    
    read -p "Enter your choice [1-4, q]: " choice
    
    case $choice in
        1)
            activate_env "$PROJECT_DIR/.pan_tilt_env" "USB Camera"
            ;;
        2)
            activate_env "$PROJECT_DIR/.zed2_complete_env" "ZED2 Camera"
            ;;
        3)
            echo -e "${YELLOW}Staying in base environment${NC}"
            ;;
        4)
            echo -e "${PURPLE}Which environment should be the default?${NC}"
            echo -e "  ${GREEN}1)${NC} USB Camera Environment"
            echo -e "  ${GREEN}2)${NC} ZED2 Camera Environment"
            read -p "Enter choice [1-2]: " default_choice
            
            case $default_choice in
                1)
                    save_choice "usb"
                    activate_env "$PROJECT_DIR/.pan_tilt_env" "USB Camera"
                    echo -e "${GREEN}USB Camera environment set as default${NC}"
                    ;;
                2)
                    save_choice "zed2"
                    activate_env "$PROJECT_DIR/.zed2_complete_env" "ZED2 Camera"
                    echo -e "${GREEN}ZED2 Camera environment set as default${NC}"
                    ;;
                *)
                    echo -e "${RED}Invalid choice${NC}"
                    ;;
            esac
            ;;
        q|Q)
            echo -e "${YELLOW}Exiting without activating environment${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            interactive_mode
            ;;
    esac
}

# Function for auto mode (uses saved preference)
auto_mode() {
    local saved_choice=$(load_choice)
    
    echo -e "${CYAN}Pan-Tilt Tracking Camera - Auto Environment Selection${NC}"
    
    case $saved_choice in
        "usb")
            echo -e "${BLUE}Loading default: USB Camera Environment${NC}"
            activate_env "$PROJECT_DIR/.pan_tilt_env" "USB Camera"
            ;;
        "zed2")
            echo -e "${BLUE}Loading default: ZED2 Camera Environment${NC}"
            activate_env "$PROJECT_DIR/.zed2_complete_env" "ZED2 Camera"
            ;;
        *)
            echo -e "${YELLOW}No default set, using interactive mode...${NC}"
            interactive_mode
            ;;
    esac
}

# Main script logic
main() {
    # Change to project directory
    cd "$PROJECT_DIR" 2>/dev/null || {
        echo -e "${RED}Error: Cannot access project directory $PROJECT_DIR${NC}"
        exit 1
    }
    
    # Check command line arguments
    case "${1:-}" in
        "--auto"|"-a")
            auto_mode
            ;;
        "--interactive"|"-i")
            interactive_mode
            ;;
        "--reset"|"-r")
            rm -f "$CONFIG_FILE"
            echo -e "${GREEN}Environment preferences reset${NC}"
            interactive_mode
            ;;
        "--help"|"-h")
            echo "Environment Selection Script"
            echo ""
            echo "Usage: $0 [option]"
            echo ""
            echo "Options:"
            echo "  -a, --auto        Use saved default environment"
            echo "  -i, --interactive Show selection menu"
            echo "  -r, --reset       Reset saved preferences"
            echo "  -h, --help        Show this help"
            echo ""
            echo "If no option is provided, interactive mode is used."
            ;;
        *)
            # Default to interactive mode
            interactive_mode
            ;;
    esac
}

# Run the main function
main "$@"