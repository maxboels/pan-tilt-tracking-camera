# ZED2 Intelligent Tracking System

A comprehensive computer vision system combining ZED2 stereo camera, YOLO object detection, 3D point cloud processing, and automated pan-tilt tracking.

## Features

- **Real-time Object Detection**: YOLOv8 integration for person/vehicle detection
- **3D Depth Sensing**: Full stereo vision with accurate distance measurements
- **Point Cloud Processing**: 3D spatial analysis and object pose estimation
- **Automated Pan-Tilt Control**: PCA9685-based servo tracking system
- **Cross-Platform**: Laptop development + Jetson Nano deployment
- **Modular Architecture**: Easy to extend and customize

## Project Structure

```
zed2_camera/
├── README.md
├── config.json                        # System configuration
├── requirements_complete.txt           # All dependencies
├── 
├── # Core Modules
├── zed_yolo_tracker.py                # Main detection/tracking logic
├── pantilt_controller.py              # Servo control system
├── pointcloud_processor.py            # 3D point cloud analysis
├── main_tracking_system.py            # Integrated system orchestrator
├── 
├── # Basic ZED Scripts (working versions)
├── zed_view_on_laptop.py              # Simple ZED viewer
├── zed_view_no_opencv.py              # OpenCV-fallback version
├── 
├── # Setup and Configuration
├── setup_laptop.sh                    # Laptop environment setup
├── setup_jetson.sh                    # Jetson Nano setup
├── 
└── wheels/
    └── pyzed-4.2-cp312-cp312-linux_x86_64.whl  # ZED SDK wheel (x86_64)
```

## Quick Start

### 1. Basic ZED Testing
```bash
# Test your ZED2 camera first
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python3 zed_view_on_laptop.py
```

### 2. Install Dependencies
```bash
# Install all requirements
pip install -r requirements_complete.txt

# Install ZED SDK wheel (laptop)
pip install wheels/pyzed-4.2-cp312-cp312-linux_x86_64.whl

# For Jetson Nano (different architecture)
cd /usr/local/zed && python3 get_python_api.py
```

### 3. Run Complete Tracking System
```bash
# Generate default configuration
python3 main_tracking_system.py --generate-config

# Run with GPU acceleration (laptop)
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python3 main_tracking_system.py

# Run on Jetson Nano
python3 main_tracking_system.py
```

## Configuration

The system uses `config.json` for settings:

```json
{
  "zed": {
    "resolution": "HD720",
    "depth_mode": "PERFORMANCE"
  },
  "yolo": {
    "model_path": "yolov8n.pt",
    "target_classes": ["person", "car", "bicycle"]
  },
  "pantilt": {
    "enabled": true,
    "pan_channel": 0,
    "tilt_channel": 1,
    "smooth_movement": true
  },
  "pointcloud": {
    "enabled": true,
    "max_depth": 10.0,
    "voxel_size": 0.05
  }
}
```

## Hardware Requirements

### Laptop (Development)
- NVIDIA GPU with CUDA support
- ZED2 stereo camera
- ZED SDK 4.2 installed
- Ubuntu 20.04+ recommended

### Jetson Nano (Deployment)  
- Jetson Nano 4GB
- ZED2 camera
- PCA9685 servo controller
- 2x servo motors (pan/tilt mechanism)
- ZED SDK 4.2 for L4T

## Module Documentation

### ZEDYOLOTracker
Core detection and tracking logic:
- Integrates ZED2 stereo vision with YOLOv8
- Provides real-time object detection with 3D coordinates
- Generates pan-tilt commands for target tracking

### PanTiltController
Servo control system:
- PCA9685-based servo control
- Smooth movement algorithms
- Safety limits and position feedback

### PointCloudProcessor
3D spatial analysis:
- Point cloud filtering and clustering
- Object pose estimation
- Occupancy mapping
- Temporal tracking

### IntegratedTrackingSystem
Main orchestrator:
- Coordinates all subsystems
- Handles configuration management
- Provides unified control interface

## Usage Examples

### Basic Object Detection
```python
from zed_yolo_tracker import ZEDYOLOTracker

tracker = ZEDYOLOTracker(target_classes=['person'])
tracker.run_detection_loop(display=True)
```

### Pan-Tilt Control
```python
from pantilt_controller import PanTiltController

controller = PanTiltController(pan_channel=0, tilt_channel=1)
controller.move_to(45, -15)  # Pan 45°, tilt -15°
```

### Point Cloud Processing
```python
from pointcloud_processor import PointCloudProcessor

processor = PointCloudProcessor()
filtered_points = processor.filter_point_cloud(raw_points)
clusters = processor.cluster_points(filtered_points)
```

## Development Workflow

### 1. Laptop Development
- Use `zed_view_on_laptop.py` to test ZED2 setup
- Develop and test algorithms with full visualization
- Use `main_tracking_system.py` for integrated testing

### 2. Jetson Deployment
- Transfer code to Jetson Nano
- Run `setup_jetson.sh` for environment setup  
- Deploy with hardware pan-tilt system

### 3. Hardware Integration
Connect PCA9685 to Jetson Nano:
```
PCA9685    Jetson Nano
VCC   ->   5V
GND   ->   GND  
SDA   ->   Pin 3 (GPIO2)
SCL   ->   Pin 5 (GPIO3)
```

Servo connections:
- Channel 0: Pan servo
- Channel 1: Tilt servo

## Troubleshooting

### OpenCV Issues
If you encounter OpenCV array recognition errors:
```bash
pip install opencv-python==4.8.1.78  # Use this specific version
```

### ZED SDK Problems
- Ensure correct architecture (x86_64 vs ARM64)
- Check `/usr/local/zed/` installation
- Verify camera permissions and connections

### PCA9685 Not Detected
```bash
# Check I2C devices
sudo i2cdetect -y -r 1

# Install I2C tools
sudo apt-get install i2c-tools
```

### Performance Optimization
- Use PERFORMANCE depth mode for speed
- Reduce point cloud resolution if needed
- Adjust YOLO model size (yolov8n vs yolov8s vs yolov8m)

## Advanced Features

### Custom Object Classes
Add your own detection classes in config:
```json
"target_classes": ["person", "bicycle", "backpack", "handbag"]
```

### 3D Tracking
Enable enhanced 3D analysis:
```json
"pointcloud": {
  "enabled": true,
  "clustering": true,
  "pose_estimation": true
}
```

### Multi-Camera Setup
The architecture supports multiple ZED cameras by instantiating multiple `ZEDYOLOTracker` objects.

## License

This project is for educational and research purposes. ZED SDK and YOLO models have their own licenses.

## Contributing

1. Test changes on laptop first
2. Ensure Jetson Nano compatibility  
3. Update documentation
4. Add unit tests where appropriate