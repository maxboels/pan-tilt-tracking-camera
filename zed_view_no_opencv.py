#!/usr/bin/env python3
import numpy as np
import cv2
import pyzed.sl as sl
import time
from PIL import Image

def colourise_depth_pil(depth_m: np.ndarray, dmin=0.3, dmax=5.0) -> np.ndarray:
    """Map 32F depth (m) to a colour image using PIL/numpy only."""
    # Create a fresh copy
    d = depth_m.astype(np.float32, copy=True)
    
    # Handle invalid values
    invalid = ~np.isfinite(d) | (d <= 0)
    d[invalid] = dmax
    d = np.clip(d, dmin, dmax)
    
    # Normalize to 0-1 range
    d_norm = (d - dmin) / (dmax - dmin)
    
    # Create RGB using a turbo-like colormap
    h, w = d_norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Turbo colormap approximation (0=blue, 0.5=green/yellow, 1=red)
    # Red channel
    rgb[:, :, 0] = np.clip(255 * (2.0 * d_norm - 0.5), 0, 255).astype(np.uint8)
    
    # Green channel  
    green = np.where(d_norm < 0.5, 
                    2.0 * d_norm,  # Ramp up 0->0.5
                    2.0 * (1.0 - d_norm))  # Ramp down 0.5->1
    rgb[:, :, 1] = (255 * green).astype(np.uint8)
    
    # Blue channel (inverted)
    rgb[:, :, 2] = (255 * (1.0 - d_norm)).astype(np.uint8)
    
    # Set invalid pixels to black
    rgb[invalid] = (0, 0, 0)
    
    return rgb

def resize_image_pil(img_array, target_height):
    """Resize image using PIL as fallback."""
    if img_array.shape[0] == target_height:
        return img_array
    
    # Calculate new width maintaining aspect ratio
    aspect = img_array.shape[1] / img_array.shape[0]
    new_width = int(target_height * aspect)
    
    # Convert to PIL, resize, convert back
    pil_img = Image.fromarray(img_array)
    pil_resized = pil_img.resize((new_width, target_height), Image.Resampling.LANCZOS)
    return np.array(pil_resized)

def main():
    zed = sl.Camera()
    init_params = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.HD720,
        depth_mode=sl.DEPTH_MODE.PERFORMANCE,
        coordinate_units=sl.UNIT.METER
    )
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {err}")

    runtime = sl.RuntimeParameters(confidence_threshold=70)
    left = sl.Mat()
    depth = sl.Mat()

    cv2.namedWindow("ZED EO + Depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZED EO + Depth", 1280, 480)

    t0, frames = time.time(), 0
    fps = 0.0

    try:
        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                frame_bgra = left.get_data()
                frame_bgr = frame_bgra[..., :3].copy()
                depth_m = depth.get_data()
                
                # Use PIL-based depth visualization
                depth_vis = colourise_depth_pil(depth_m, dmin=0.3, dmax=5.0)

                # Make same height and stack - using PIL for resize as fallback
                h = min(frame_bgr.shape[0], depth_vis.shape[0])
                
                try:
                    # Try OpenCV resize first
                    frame_resized = cv2.resize(frame_bgr, (int(frame_bgr.shape[1]*h/frame_bgr.shape[0]), h))
                    depth_resized = cv2.resize(depth_vis, (int(depth_vis.shape[1]*h/depth_vis.shape[0]), h))
                except:
                    # Fallback to PIL resize
                    print("Using PIL resize fallback")
                    frame_resized = resize_image_pil(frame_bgr, h)
                    depth_resized = resize_image_pil(depth_vis, h)
                
                vis = np.hstack([frame_resized, depth_resized])

                # FPS
                frames += 1
                dt = time.time() - t0
                if dt >= 1.0:
                    fps = frames / dt
                    t0, frames = time.time(), 0
                
                try:
                    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                except:
                    # Skip text overlay if OpenCV text fails too
                    pass

                # Show centre depth
                h2, w2 = frame_bgr.shape[:2]
                u, v = w2 // 2, h2 // 2
                Z = float(depth_m[v, u])
                if np.isfinite(Z) and Z > 0:
                    try:
                        cv2.putText(vis, f"Centre depth: {Z:.2f} m", (10, 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    except:
                        pass

                try:
                    cv2.imshow("ZED EO + Depth", vis)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        break
                except:
                    print("OpenCV display failed, but depth processing working!")
                    print(f"Frame shape: {vis.shape}, dtype: {vis.dtype}")
                    break
                    
            else:
                time.sleep(0.002)
    finally:
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()