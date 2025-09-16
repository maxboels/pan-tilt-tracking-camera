# Pan-Tilt Tracking Camera

A computer vision system for tracking people using a USB camera and controlling a pan-tilt servo mechanism with YOLOv8-based object detection.

![Pan-Tilt Tracking Camera](captures/Pan-Tilt%20Tracking%20Camera_screenshot_12.09.2025.png)

## Features

- **YOLO Person Detection**: Uses YOLOv8 for robust and fast person detection
- **Real-time Tracking**: Smooth tracking with position smoothing and dead-zone control
- **Pan-Tilt Control**: Arduino-based servo control with inverted pan servo support
- **USB Camera Support**: Works with standard UVC USB cameras
- **Performance Logging**: Comprehensive logging system for evaluation and analysis

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Installation](#installation)
3. [Inference](#inference)
4. [Evaluation](#evaluation)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

## Hardware Requirements

- **Camera**: USB 2.0 UVC camera (tested with U20CAM-1080P-1)
- **Servos**: 2x MG996R servos for pan and tilt movement
- **Servo Controller**: 1x PCA9685 16-channel servo controller board
- **Arduino**: Arduino compatible with the Servo library (tested with Arduino Uno/Nano)
- **Computer**: Linux system (tested on Ubuntu and NVIDIA Jetson Nano)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pan-tilt-tracking-camera.git
   cd pan-tilt-tracking-camera
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```
   This creates a Python virtual environment and installs all dependencies.

3. **Upload Arduino code**:
   Upload the `src/arduino_servo_bridge.ino` to your Arduino board using the Arduino IDE.

4. **Activate the environment**:
   ```bash
   source venv/bin/activate
   ```

## Inference

The main application uses YOLOv8 to detect people in the camera feed and controls the pan-tilt servo mechanism to track them.

### Basic Command

```bash
# Run with default settings
python main.py
```

### Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--config`, `-c` | Path to configuration file (default: config/config.json) |
| `--camera` | Camera index override (default: 0) |
| `--model` | Custom YOLO model path |
| `--experiment`, `-e` | Name for this experiment run (for logging) |
| `--eval` | Automatically run evaluation when quitting |

### Examples

```bash
# Use a specific camera
python main.py --camera 1

# Use a different YOLO model
python main.py --model yolov8s.pt

# Name your experiment and enable auto-evaluation
python main.py --experiment my_tracking_test --eval
```

### Interactive Controls

| Key | Function |
|-----|----------|
| `q` | Quit the application |
| `r` | Reset tracking (clears tracking history) |
| `t` | Toggle tracking on/off |
| `c` | Center servos (move to neutral position) |
| `s` | Save current frame to captures folder |
| `space` | Pause/resume video feed |

## Evaluation

The system includes performance logging and evaluation tools to analyze tracking results.

### Running Evaluations

```bash
# Analyze the most recent experiment
python evals/analyze_tracking_logs.py

# Analyze a specific experiment
python evals/analyze_tracking_logs.py --experiment experiment_20250916_143411

# Legacy log format
python evals/analyze_tracking_logs.py --log logs/tracking_run_20250913_123045.log
```

### Auto-evaluation

Add the `--eval` flag to automatically run evaluation when quitting the application:

```bash
python main.py --experiment my_test_run --eval
```

### Synthetic Testing

The system includes synthetic testing capabilities for reproducible experiments:

```bash
# Run synthetic benchmark with default circular movement pattern
python evals/synthetic_benchmark.py

# Specify movement pattern
python evals/synthetic_benchmark.py --pattern linear

# Configure duration and experiment name
python evals/synthetic_benchmark.py --duration 30 --experiment linear_test
```

### Visualizing Results

```bash
# Visualize the most recent benchmark
python evals/synthetic_visualization.py

# Visualize a specific experiment
python evals/synthetic_visualization.py --experiment benchmark_20250913_123045

# Generate animation or video
python evals/synthetic_visualization.py --animation
python evals/synthetic_visualization.py --video
```

## Configuration

The system is configured via JSON files:

- `config/config.json` - Main configuration
- `config/calibration.json` - Camera-servo calibration data

Key configuration parameters:

```json
{
  "camera": {
    "index": 0,
    "resolution": [1920, 1080],
    "fps": 30
  },
  "servo": {
    "port": "/dev/ttyACM0",
    "baudrate": 115200,
    "inverted_pan": true
  },
  "tracking": {
    "model_path": "yolov8n.pt",
    "confidence_threshold": 0.5,
    "dead_zone": 20
  }
}
```

## Troubleshooting

### Camera Issues
- Ensure the camera is properly connected and recognized
- Check camera permissions (`ls -l /dev/video*`)
- Verify camera index with `v4l2-ctl --list-devices`

### Servo Issues
- Verify Arduino connection (`ls -l /dev/ttyACM*` or `/dev/ttyUSB*`)
- Check servo wiring and power supply
- Run `python evals/servo_diagnostic.py` to test servo functionality

### Detection Issues
- Ensure YOLO model file exists
- Try a different model (e.g., `--model yolov8s.pt` for better accuracy)
- Adjust confidence threshold in config file