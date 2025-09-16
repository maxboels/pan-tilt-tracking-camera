# Pan-Tilt Tracking Camera

A computer vision system for tracking people using a USB camera and controlling a pan-tilt servo mechanism with YOLOv8-based object detection.

![Pan-Tilt Tracking Camera](captures/Pan-Tilt%20Tracking%20Camera_screenshot_12.09.2025.png)

## Features

- **YOLO Person Detection**: Uses YOLOv8 for robust and fast person detection
- **Real-time Tracking**: Smooth tracking with Kalman filtering and position smoothing
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
# Run with default settings (recording and Kalman filtering enabled)
python main.py

# Run without tracking (camera will show detections but won't move servos)
python main.py --no-tracking

# Run without video recording
python main.py --no-record
```

### Command-line Arguments

| Argument | Description |
|----------|-------------|
| `--config`, `-c` | Path to configuration file (default: config/config.json) |
| `--camera` | Camera index override (default: 0) |
| `--model` | Custom YOLO model path |
| `--experiment`, `-e` | Name for this experiment run (for logging) |
| `--eval` | Automatically run evaluation when quitting |
| `--no-tracking`, `-n` | Start with tracking disabled (servos won't move) |
| `--no-record`, `-nr` | Disable video recording (enabled by default) |
| `--mode`, `-m` | Tracking mode: 'surveillance' (keeps person in scene) or 'turret' (aims at center of bounding box) |
| `--no-kalman`, `-nk` | Disable Kalman filtering (enabled by default) |
| `--no-compensation`, `-nc` | Disable motion compensation for preventing feedback loops (enabled by default) |

### Examples

```bash
# Run with default settings (recording enabled, surveillance mode)
python main.py

# Use turret mode for precision targeting (recording still enabled)
python main.py --mode turret

# Use standard surveillance mode
python main.py --mode surveillance

# Run with a specific experiment name (for organization)
python main.py --experiment tracking_demo_1

# Record without tracking (useful for creating test footage)
python main.py --no-tracking

# Track in turret mode and run evaluation when finished
# Run with all options (recording and Kalman filtering enabled by default)
python main.py --mode turret --eval
```

### Tracking Modes

The system supports two distinct tracking modes optimized for different use cases:

1. **Surveillance Mode** (Default): 
   - Designed to keep the person in the scene with stable, smooth camera movements
   - Uses a longer history buffer (10 positions) for position smoothing
   - Gradual weight distribution provides stable tracking with minimal jitter
   - Best for general surveillance and monitoring applications

2. **Turret Mode**:
   - Designed for precise targeting at the center of the detected bounding box
   - Uses a shorter history buffer (3 positions) for more immediate response
   - Applies exponential weighting to heavily favor recent positions
   - Best for applications requiring precise aiming like laser pointer tracking

Switch between modes using the `--mode` command line argument:
```bash
python main.py --mode surveillance  # Default mode
python main.py --mode turret        # Precision targeting mode
```

### Kalman Filtering

The system supports Kalman filtering for improved tracking performance:

- Predicts target positions based on estimated velocity
- Maintains tracking during brief occlusions
- Reduces jitter while maintaining responsiveness
- Automatically tunes parameters based on the selected tracking mode
- Turret mode uses more responsive filter settings
- Surveillance mode uses smoother filter settings

Kalman filtering is enabled by default. If needed, you can disable it with the `--no-kalman` flag:
```bash
python main.py  # Kalman filtering is enabled by default
python main.py --mode turret  # Precision targeting with Kalman filtering
python main.py --no-kalman  # Disable Kalman filtering if needed
```

### Motion Compensation

The system includes advanced motion compensation to prevent feedback loops that can cause drift:

- Distinguishes between real target motion and apparent motion caused by camera movement
- Prevents system instability when tracking stationary targets
- Gradually builds confidence in stationary targets to further reduce drift
- Compatible with both tracking modes and Kalman filtering

Motion compensation is enabled by default. If needed, you can disable it with the `--no-compensation` flag:
```bash
python main.py  # Motion compensation is enabled by default
python main.py --mode turret  # Precision targeting with motion compensation
python main.py --no-compensation  # Disable motion compensation if needed
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