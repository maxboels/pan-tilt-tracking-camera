# Pan-Tilt Tracking Camera

A clean, organized computer vision system for tracking people using a USB camera and controlling a pan-tilt mechanism with YOLO-based detection.

## Features

- **YOLO Person Detection**: Uses YOLOv8 for robust and fast person detection
- **Real-time Tracking**: Smooth tracking with position smoothing and dead-zone control
- **Pan-Tilt Control**: Arduino-based servo control with inverted pan servo support
- **USB Camera Support**: Works with standard UVC USB cameras
- **Calibration**: Camera-servo calibration system for accurate targeting
- **Clean Architecture**: Well-organized modular codebase

## Hardware Requirements

- **Camera**: USB 2.0 UVC camera (tested with U20CAM-1080P-1)
- **Servos**: Pan-tilt servo mechanism with Arduino controller
- **Computer**: Linux system (tested on Ubuntu)

## Quick Start

### 1. Setup Environment

```bash
./setup.sh
```

### 2. Activate Environment

```bash
source .pan_tilt_env/bin/activate
```

### 3. Run Tracking System

```bash
python main.py
```

**Controls:**
- `q` - Quit
- `r` - Reset tracking
- `t` - Toggle tracking on/off
- `c` - Center servos
- `s` - Save current frame

### 4. Calibration (Optional)

```bash
python examples/calibration_tool.py
```

## Project Structure

```
pan-tilt-tracking-camera/
├── main.py                    # Main application
├── setup.sh                   # Setup script
├── requirements.txt           # Python dependencies
│
├── src/                       # Core modules
│   ├── yolo_tracker.py       # YOLO-based object tracking
│   ├── usb_camera.py         # USB camera interface
│   ├── servo_controller.py   # Arduino servo control
│   └── calibration.py        # Camera-servo calibration
│
├── config/                    # Configuration files
│   ├── config.json           # Main configuration
│   └── calibration.json      # Calibration data (auto-generated)
│
├── examples/                  # Example scripts
│   └── calibration_tool.py   # Interactive calibration tool
│
├── captures/                  # Saved frames
├── logs/                      # Log files
└── yolov8n.pt                # YOLO model (auto-downloaded)
```

## Configuration

Edit `config/config.json` to customize settings:

```json
{
  "camera": {
    "index": 0,
    "resolution": [1920, 1080],
    "fps": 30
  },
  "servo": {
    "port": "/dev/ttyUSB0",
    "baudrate": 115200,
    "inverted_pan": true
  },
  "tracking": {
    "model_path": "yolov8n.pt",
    "confidence_threshold": 0.5,
    "dead_zone": 50
  }
}
```

## Command Line Options

```bash
python main.py --help
```

Available options:
- `--config` - Configuration file path
- `--camera` - Camera index override
- `--model` - YOLO model path override

## Development

### Testing Individual Components

```bash
# Test YOLO tracker
python src/yolo_tracker.py

# Test USB camera
python src/usb_camera.py

# Calibration tool
python examples/calibration_tool.py
```

### Key Features

#### YOLO Detection
- Uses YOLOv8 for fast, accurate person detection
- Configurable confidence thresholds
- Optimized for real-time performance

#### Servo Control
- Arduino-based control with serial communication
- Support for inverted pan servo (common requirement)
- Smooth movement with speed limiting
- Position feedback and tracking

#### Tracking System
- Position smoothing with weighted averaging
- Dead-zone control to prevent jittery movements
- Visual feedback with tracking history
- FPS monitoring and performance tracking

## Troubleshooting

### Camera Issues
```bash
# List available cameras
ls /dev/video*

# Test camera directly
python src/usb_camera.py
```

### Servo Issues
```bash
# Check serial ports
ls /dev/ttyUSB*

# Test servo controller
python -c "from src.servo_controller import ArduinoServoController; c=ArduinoServoController(); print(c.connect())"
```

### Performance Issues
- Reduce camera resolution in config
- Lower FPS target
- Use smaller YOLO model (yolov8n.pt vs yolov8s.pt)
- Ensure good lighting conditions

## Migration Notes

This is a cleaned and reorganized version of the original codebase that:

- **Removed**: ZED2 camera support, debug scripts, overcomplicated tracking systems
- **Added**: Clean modular architecture, YOLO-based detection, better configuration
- **Kept**: USB camera support, Arduino servo control, inverted pan servo feature
- **Improved**: Code organization, documentation, ease of use

The system focuses on the essential functionality needed for reliable person tracking with a USB camera and pan-tilt servos.
