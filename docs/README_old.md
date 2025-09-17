# Pan-Tilt Tracking Camera

A computer vision system for tracking people using a USB camera and controlling a pan-tilt mechanism.

## Hardware Requirements

- **Camera**: U20CAM-1080P-1 (1080P USB 2.0 UVC Camera with 130° Wide Angle)
- **Computer**: Linux laptop (tested on Ubuntu)
- **Future**: NVIDIA Jetson Nano Dev Kit support planned

## Quick Start

### 1. Set up the virtual environment

```bash
chmod +x setup_usb_cam_env.sh
./setup_usb_cam_env.sh
```

### 2. Activate the environment

```bash
source .pan_tilt_env/bin/activate
```

### 3. Test the camera

```bash
python usb_camera.py
```

This will open a test window showing the camera feed. Press 'q' to quit or 's' to save a test frame.

### 4. Run the tracking system

```bash
python main_usb_tracking.py
```

**Controls:**
- Press 'q' to quit
- Press 's' to save current frame
- Press 'r' to reset tracking

### Command line options:

```bash
python main_usb_tracking.py --camera 0 --fps 30
```

- `--camera`: Camera index (default: 0 for first USB camera)
- `--fps`: Target frames per second (default: 30)

## Project Migration from ZED2

This project was previously called `zed2_camera` and used a ZED2 stereo camera. It has been updated to:

1. **New name**: `pan-tilt-tracking-camera`
2. **New camera**: U20CAM-1080P-1 USB camera
3. **New environment**: `.pan_tilt_env` (replaces `.zed2_complete_env`)
4. **Simplified setup**: No ZED SDK required

## Features

- **Person Detection**: Uses MediaPipe Pose for robust person tracking
- **Real-time Tracking**: Tracks the nose landmark for stable targeting
- **Pan-Tilt Error Calculation**: Calculates movement needed to center target
- **Dead Zone**: Configurable dead zone to prevent jittery movements
- **Performance Monitoring**: Real-time FPS display
- **Visual Feedback**: Shows tracking point, center crosshair, and error values

## Technical Details

### Camera Specifications
- **Resolution**: 1920x1080 (Full HD)
- **Field of View**: 130° wide angle
- **Interface**: USB 2.0 UVC (plug & play)
- **Frame Rate**: Up to 30 FPS

### Tracking Algorithm
1. Capture frame from USB camera
2. Convert to RGB for MediaPipe processing
3. Detect person pose landmarks
4. Track nose landmark as target point
5. Calculate pan/tilt error from frame center
6. Apply dead zone to prevent micro-movements
7. Send control commands (hardware integration pending)

## Files Structure

```
pan-tilt-tracking-camera/
├── setup_usb_cam_env.sh        # Environment setup script
├── requirements_usb_cam.txt    # Python dependencies
├── usb_camera.py              # USB camera interface
├── main_usb_tracking.py       # Main tracking application
└── README.md                  # This file
```

## Development Notes

- **Current Status**: Camera capture and person tracking working
- **Next Steps**: Integrate servo control hardware
- **Future Migration**: Move to NVIDIA Jetson Nano for deployment

## Troubleshooting

### Camera not detected
```bash
# List available cameras
ls /dev/video*

# Check camera info
v4l2-ctl --list-devices
```

### Low FPS or performance issues
- Reduce resolution in `USBCamera` initialization
- Lower the target FPS
- Ensure good lighting conditions
- Close other applications using the camera

### Permission issues
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Log out and log back in
```