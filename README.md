# Pan-Tilt Tracking Camera

A clean, organized computer vision system for tracking people using a USB camera and controlling a pan-tilt servo mechanism with YOLO-based detection.

![Pan-Tilt Tracking Camera](captures/Pan-Tilt%20Tracking%20Camera_screenshot_12.09.2025.png)

## Features

- **YOLO Person Detection**: Uses YOLOv8 for robust and fast person detection
- **Real-time Tracking**: Smooth tracking with position smoothing and dead-zone control
- **Pan-Tilt Control**: Arduino-based servo control with inverted pan servo support
- **USB Camera Support**: Works with standard UVC USB cameras
- **Calibration**: Camera-servo calibration system for accurate targeting
- **Clean Architecture**: Well-organized modular codebase
- **Performance Logging**: Comprehensive logging system for experimental evaluation and analysis
- **Synthetic Benchmarking**: Generate synthetic movement data for reproducible experiments
- **Performance Visualization**: Detailed visualizations comparing ground truth vs tracked positions

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Installation and Setup](#installation-and-setup)
3. [Quick Start Guide](#quick-start-guide)
4. [Real-time Person Tracking](#real-time-person-tracking)
5. [Synthetic Testing](#synthetic-testing)
   - [Benchmark Testing](#running-synthetic-benchmarks)
   - [Hardware-in-the-loop Testing](#hardware-in-the-loop-testing)
   - [Visualizing Results](#visualizing-benchmark-results)
6. [Configuration](#configuration)
7. [Project Structure](#project-structure)
8. [Troubleshooting](#troubleshooting)

## Hardware Requirements

- **Camera**: USB 2.0 UVC camera (tested with U20CAM-1080P-1)
  - High-Speed USB 2.0 UVC camera module with driver-free support
  - 1920x1080@30fps output (YUY2/MJPEG) 
  - 130° Wide-Angle Optics with 103°(H)/130°(D) FOV
  - M12 lens thread for optical customization
  - 32x32mm PCB with 4x M2 mounting holes
  - Operating temperature: -20°C~70°C
- **Servos**: 
  - 2x MG996R servos for pan and tilt movement
  - Pan-tilt servo mechanism with Arduino controller (running `arduino_servo_bridge.ino`)
- **Servo Controller**:
  - 1x PCA9685 16-channel servo controller board
- **Arduino**: Any Arduino compatible with the Servo library (tested with Arduino R3/Nano/Uno)
- **Computer**: Linux system (tested on Ubuntu and NVIDIA Jetson Nano)
- **Connectivity**: USB ports for camera and Arduino

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/maxboels/pan-tilt-tracking-camera.git
   cd pan-tilt-tracking-camera
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```
   This creates a Python virtual environment and installs all dependencies.

3. **Upload Arduino code**:
   Upload the `src/arduino_servo_bridge.ino` to your Arduino board using the Arduino IDE.

4. **Connect Hardware**:
   - Connect the USB camera to your computer
   - Connect the Arduino to your computer
   - Connect servo motors to the Arduino (see pin configuration in the Arduino code)

## Quick Start Guide

1. **Activate the environment**:
   ```bash
   source .pan_tilt_env/bin/activate
   ```

2. **Run the tracking system**:
   ```bash
   python main.py
   ```

## Real-time Person Tracking

The main application uses YOLOv8 to detect people in the camera feed and controls the pan-tilt servo mechanism to keep them centered in the frame.

### Basic Usage

```bash
# Run with default settings
python main.py

# Specify a config file
python main.py --config config/my_custom_config.json

# Name your experiment for better organization
python main.py --experiment my_experiment_name

# Override camera index
python main.py --camera 1

# Specify a custom YOLO model
python main.py --model yolov8s.pt
```

### Interactive Controls

During operation, the following keyboard controls are available:

| Key | Function |
|-----|----------|
| `q` | Quit the application |
| `r` | Reset tracking (clears tracking history) |
| `t` | Toggle tracking on/off |
| `c` | Center servos (move to neutral position) |
| `s` | Save current frame to captures folder |
| `space` | Pause/resume video feed |

### Display Information

The on-screen display shows:
- FPS counter
- Detection confidence
- Tracking status
- Current servo positions
- Pixel error from center
- Tracking history (green trails)

### 4. Calibration (Optional)

```bash
python examples/calibration_tool.py
```

## Performance Logging and Analysis

The system includes a comprehensive performance logging system that records detailed metrics for each tracking run, allowing for consistent evaluation and experimental analysis.

### Logged Metrics

For each frame, the system logs:
- Timestamp
- Camera center coordinates
- Detected object bounding box coordinates
- Center point of detected objects
- Servo commands sent
- Distance error between target and camera center
- Processing time and performance metrics

### Testing the Logger

```bash
python examples/test_tracking_logger.py
```

### Analyzing Log Data

After a tracking session, you can analyze the log data using the provided analysis tool:

```bash
python examples/analyze_tracking_logs.py
```

This will automatically load the most recent log file and generate visualizations showing:
- Distance error over time
- Servo position tracking
- Target position heatmap
- Processing time analysis
- Error distribution

You can also specify a particular log file:

```bash
python examples/analyze_tracking_logs.py --experiment experiment_20250913_190229
```

Or for legacy log files:

```bash
python examples/analyze_tracking_logs.py --log logs/tracking_run_20250913_123045.log
```

## Synthetic Testing

The system includes comprehensive synthetic testing capabilities that allow for reproducible experiments and evaluation of the tracking system. This section explains the different synthetic testing tools available and how to use them effectively.

### Overview of Synthetic Testing Tools

The synthetic testing suite consists of three main components:

1. **Synthetic Benchmarking** (`examples/synthetic_benchmark.py`): Evaluates tracking algorithms with synthetic data, without hardware
2. **Hardware-in-the-loop Testing** (`examples/synthetic_servo_test.py`): Tests real servo hardware with synthetic detection data
3. **Results Visualization** (`examples/synthetic_visualization.py`): Analyzes and visualizes test results with detailed metrics

### Running Synthetic Benchmarks

Synthetic benchmarking allows you to evaluate the tracking algorithm's performance with predefined movement patterns without requiring actual hardware.

```bash
# Run with default circular movement pattern
python examples/synthetic_benchmark.py

# Specify a different movement pattern
python examples/synthetic_benchmark.py --pattern linear

# Set experiment duration and name
python examples/synthetic_benchmark.py --duration 30 --experiment linear_test

# Combine multiple parameters
python examples/synthetic_benchmark.py --pattern zigzag --duration 60 --experiment zigzag_long_test
```

#### Available Movement Patterns

The system supports multiple predefined movement patterns to test different tracking scenarios:

| Pattern | Description | Best For Testing |
|---------|-------------|------------------|
| `linear` | Straight line movement | Simple tracking, initial acquisition |
| `circular` | Continuous circular motion | Sustained tracking, servo speed limits |
| `random` | Random walk movement | Unpredictable targets, robustness |
| `zigzag` | Back-and-forth pattern | Direction changes, tracking stability |
| `spiral` | Outward spiral motion | Variable speed tracking |

#### Benchmark Command Options

```bash
python examples/synthetic_benchmark.py --help
```

Full options list:
```
--pattern PATTERN     Movement pattern (linear, circular, zigzag, random, spiral)
--duration DURATION   Test duration in seconds (default: 60)
--experiment NAME     Experiment name for logging (default: timestamp-based)
--fps FPS             Simulation frames per second (default: 30)
--width WIDTH         Frame width in pixels (default: 1920)
--height HEIGHT       Frame height in pixels (default: 1080)
--radius RADIUS       Radius for circular/spiral patterns (default: 400)
--speed SPEED         Movement speed factor (default: 1)
--clockwise           Use clockwise motion for circular pattern (default)
--counter-clockwise   Use counter-clockwise motion for circular pattern
--direction DIR       Direction for linear pattern (left_to_right, diagonal_down, etc.)
```

### Visualizing Benchmark Results

After running a benchmark test, you can visualize the results to analyze tracking performance:

```bash
# Visualize the most recent benchmark results
python examples/synthetic_visualization.py

# Visualize a specific experiment by name
python examples/synthetic_visualization.py --experiment benchmark_20250913_123045

# Create an animated visualization
python examples/synthetic_visualization.py --animation

# Generate a video visualization
python examples/synthetic_visualization.py --video

# Export data to CSV for further analysis
python examples/synthetic_visualization.py --export
```

#### Visualization Output

The visualization tool generates several plots and metrics:

1. **Path Comparison**: Shows ground truth vs. tracked positions on a 2D plot
2. **Position Tracking**: X and Y coordinates plotted over time
3. **Error Analysis**: Tracking error over time and error distribution
4. **Performance Metrics**:
   - Mean Pixel Error (MPE)
   - Root Mean Square Error (RMSE)
   - Maximum Error
   - Average Tracking Rate (%)
   - Servo Response Time

#### Interpreting Results

The visualization outputs provide insights into tracking performance:

- **Low MPE Values**: Better tracking accuracy (typically < 10px is excellent)
- **Consistent Error**: More stable tracking than variable error
- **Tracking Rate**: Percentage of frames where the target was successfully tracked
- **Response Lag**: Delay between target movement and servo response

### Hardware-in-the-loop Testing

While synthetic benchmarks evaluate the tracking algorithm, hardware-in-the-loop testing evaluates how well the real servo hardware can follow synthetic target movements:

```bash
# Test with default circular pattern
python examples/synthetic_servo_test.py

# Specify Arduino port
python examples/synthetic_servo_test.py --port /dev/ttyACM0

# Test with different movement pattern
python examples/synthetic_servo_test.py --pattern zigzag

# Combined options
python examples/synthetic_servo_test.py --pattern linear --duration 45 --experiment hardware_linear_test
```

#### Hardware Test Options

The hardware test accepts the same movement pattern options as the synthetic benchmark, plus hardware-specific options:

```
--port PORT           Serial port for Arduino servo controller (default: /dev/ttyUSB0)
--baudrate RATE       Serial baudrate (default: 115200)
--inverted            Set if pan servo is inverted (default: true)
--non-inverted        Set if pan servo is not inverted
```

#### Test Procedure

1. Connect your Arduino with the `arduino_servo_bridge.ino` sketch uploaded
2. Run the hardware test with your chosen movement pattern
3. The system will:
   - Generate synthetic object positions
   - Calculate servo commands based on those positions
   - Send real commands to your hardware
   - Log the actual servo positions vs. synthetic object positions
4. After the test, visualize the results as with regular benchmarks

## Experimental Evaluation Protocol

The system provides a structured protocol for experimental evaluation of tracking performance, allowing for:
1. Comparing different tracking algorithms or parameters
2. Measuring performance across different hardware configurations
3. Analyzing tracking stability and accuracy
4. Identifying system bottlenecks

### Movement Pattern Evaluation

Each movement pattern tests different aspects of the tracking system:

| Pattern | What It Tests | Expected Performance |
|---------|---------------|----------------------|
| Linear | Basic acquisition and tracking | Very high accuracy (2-5px error) |
| Zigzag | Direction change response | Good accuracy (5-10px error) |
| Circular | Continuous motion tracking | Moderate accuracy (10-30px error) |
| Random | Unpredictable target handling | Variable (depends on randomness) |
| Spiral | Variable speed tracking | Decreasing accuracy with radius |

### Understanding Benchmark Metrics

The system reports several key metrics to evaluate tracking performance:

1. **Mean Pixel Error (MPE)**: Average distance in pixels between the ground truth and tracked position
   - `< 5px`: Excellent tracking
   - `5-15px`: Good tracking
   - `15-30px`: Fair tracking
   - `> 30px`: Poor tracking that needs improvement

2. **Root Mean Square Error (RMSE)**: Emphasizes larger errors
   - Higher than MPE indicates inconsistent tracking with occasional large errors

3. **Tracking Rate**: Percentage of frames where tracking was maintained
   - `> 95%`: Excellent
   - `90-95%`: Good
   - `80-90%`: Fair
   - `< 80%`: Poor, needs investigation

4. **Servo Response Time**: Delay between target movement and servo response
   - Typically 100-300ms depending on hardware

### Experiment Organization

Each experiment is organized in its own subfolder within the `logs/` directory:

```
logs/
  └── experiment_name_20250913_123045/
      ├── tracking_data.log       # CSV log file with per-frame metrics
      ├── system_config.json      # System configuration snapshot
      ├── evaluation_data.json    # Benchmark results data
      ├── tracking_comparison.png # Performance visualization
      └── frames/                 # Captured frames during the experiment
          ├── frame_1757780945.jpg
          └── frame_1757780947.jpg
```

### Interpreting Synthetic Test Results

When analyzing benchmark results, consider:

1. **Pattern Complexity**: More complex patterns (circular, random) naturally have higher error
2. **Hardware Limitations**: Servo speed and acceleration limits affect tracking of fast-moving targets
3. **Algorithm Performance**: How well the tracking algorithm predicts and follows the target
4. **Error Patterns**: Look for systematic errors vs. random fluctuations
   - Consistent lag indicates prediction could help
   - Random spikes may indicate calibration issues
   - Growing error may indicate cumulative problems

Use the visualization tools to identify specific failure modes and improvement opportunities in your tracking system.

## Project Structure

```
pan-tilt-tracking-camera/
├── main.py                    # Main application
├── setup.sh                   # Setup script
├── requirements.txt           # Python dependencies
│
├── src/                       # Core modules
│   ├── yolo_tracker.py        # YOLO-based object tracking
│   ├── usb_camera.py          # USB camera interface
│   ├── servo_controller.py    # Arduino servo control
│   ├── calibration.py         # Camera-servo calibration
│   ├── tracking_logger.py     # Tracking metrics logger
│   └── synthetic_detection.py # Synthetic detection generator
│
├── config/                    # Configuration files
│   ├── config.json            # Main configuration
│   └── calibration.json       # Calibration data (auto-generated)
│
├── examples/                  # Example scripts
│   ├── calibration_tool.py    # Interactive calibration tool
│   ├── analyze_tracking_logs.py # Log analysis tool
│   ├── synthetic_benchmark.py # Synthetic tracking benchmark
│   └── synthetic_visualization.py # Benchmark visualization
│
├── captures/                  # Saved frames
├── logs/                      # Log files and experiment data
│   └── experiment_name_20250913_123045/ # Experiment folders
│       ├── tracking_data.log  # Tracking metrics
│       ├── evaluation_data.json # Benchmark data
│       └── tracking_comparison.png # Visualization output
│
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

## Advanced Tracking Techniques

The tracking system supports several advanced techniques that can be implemented to improve performance:

### Kalman Filtering

For improved trajectory prediction, especially with erratic movements:

```bash
# Enable Kalman filtering in the config
python main.py --config config/kalman_config.json
```

### Motion Prediction

The system can use predictive algorithms to anticipate target movement:

```bash
# Enable motion prediction
python main.py --predict
```

### Multi-Target Tracking

For tracking multiple people in the scene:

```bash
# Enable multi-target tracking
python main.py --multi-target
```

## System Components

### YOLO Detection Module
- Uses YOLOv8 for fast, accurate person detection
- Configurable confidence thresholds and classes
- Optimized for real-time performance on various hardware
- Automatic model downloading and caching

### Servo Control System
- Arduino-based control with serial communication
- Support for inverted pan servo (common requirement)
- Smooth movement with speed limiting and acceleration control
- Position feedback and tracking with error correction
- Hardware abstraction layer for different servo types

### Tracking Algorithms
- Position smoothing with weighted averaging
- Dead-zone control to prevent jittery movements
- Visual feedback with tracking history visualization
- FPS monitoring and performance tracking
- Optional trajectory prediction

### Synthetic Testing Framework
- Multiple movement pattern generators (linear, circular, zigzag, etc.)
- Hardware-in-the-loop testing capability
- Comprehensive performance metrics and visualization
- Experiment management and data organization

### Calibration System
- Interactive camera-servo calibration tool
- Persistent calibration storage
- Automatic mapping between pixel coordinates and servo angles
- Compensation for camera distortion and servo non-linearity

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
