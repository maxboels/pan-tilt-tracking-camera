#!/usr/bin/env python3
import numpy as np
import cv2
import pyzed.sl as sl
import time

def colourise_depth(depth_m: np.ndarray, dmin=0.3, dmax=5.0) -> np.ndarray:
    """Map 32F depth (m) to a colour image. Invalids -> black."""
    # Make a completely fresh numpy array to avoid any OpenCV compatibility issues
    d = np.array(depth_m, dtype=np.float32, copy=True, order='C')
    
    # Handle invalid values
    invalid = ~np.isfinite(d) | (d <= 0)
    d[invalid] = dmax
    d = np.clip(d, dmin, dmax)
    
    # Normalize to 0-255 range
    d_norm = ((d - dmin) / (dmax - dmin) * 255.0)
    d_norm = d_norm.astype(np.uint8)
    
    # Create a completely fresh array for OpenCV
    d_inverted = 255 - d_norm
    
    # Force create a brand new array with explicit memory layout
    opencv_ready = np.empty_like(d_inverted, dtype=np.uint8, order='C')
    opencv_ready[:] = d_inverted
    
    print(f"OpenCV ready array - type: {type(opencv_ready)}, dtype: {opencv_ready.dtype}")
    print(f"Shape: {opencv_ready.shape}, strides: {opencv_ready.strides}")
    print(f"Contiguous: {opencv_ready.flags['C_CONTIGUOUS']}, owns data: {opencv_ready.flags['OWNDATA']}")
    
    try:
        # Try with the recreated array
        cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        depth_color = cv2.applyColorMap(opencv_ready, cmap)
        depth_color[invalid] = (0, 0, 0)
        return depth_color
    except Exception as e:
        print(f"Still failed: {e}")
        
        # Ultimate fallback - manual colormap using numpy
        print("Using manual colormap fallback...")
        
        # Create RGB manually using a simple colormap
        h, w = opencv_ready.shape
        depth_color = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Simple blue->green->red colormap
        normalized = opencv_ready.astype(np.float32) / 255.0
        
        # Blue channel (high at low values)
        depth_color[:, :, 0] = (255 * (1 - normalized)).astype(np.uint8)
        # Green channel (peak in middle)
        depth_color[:, :, 1] = (255 * np.sin(normalized * np.pi)).astype(np.uint8)
        # Red channel (high at high values)  
        depth_color[:, :, 2] = (255 * normalized).astype(np.uint8)
        
        depth_color[invalid] = (0, 0, 0)
        return depth_color

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
                
                # Only print debug info on first few frames
                if frames < 2:
                    depth_vis = colourise_depth(depth_m, dmin=0.3, dmax=5.0)
                else:
                    # Simplified version for performance after debug
                    d = np.array(depth_m, dtype=np.float32, copy=True, order='C')
                    invalid = ~np.isfinite(d) | (d <= 0)
                    d[invalid] = 5.0
                    d = np.clip(d, 0.3, 5.0)
                    d_norm = ((d - 0.3) / (5.0 - 0.3) * 255.0).astype(np.uint8)
                    d_inverted = 255 - d_norm
                    
                    opencv_ready = np.empty_like(d_inverted, dtype=np.uint8, order='C')
                    opencv_ready[:] = d_inverted
                    
                    try:
                        cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
                        depth_vis = cv2.applyColorMap(opencv_ready, cmap)
                        depth_vis[invalid] = (0, 0, 0)
                    except:
                        # Manual colormap fallback
                        h, w = opencv_ready.shape
                        depth_vis = np.zeros((h, w, 3), dtype=np.uint8)
                        normalized = opencv_ready.astype(np.float32) / 255.0
                        depth_vis[:, :, 0] = (255 * (1 - normalized)).astype(np.uint8)
                        depth_vis[:, :, 1] = (255 * np.sin(normalized * np.pi)).astype(np.uint8)
                        depth_vis[:, :, 2] = (255 * normalized).astype(np.uint8)
                        depth_vis[invalid] = (0, 0, 0)

                # Make same height and stack
                h = min(frame_bgr.shape[0], depth_vis.shape[0])
                def rh(img): return cv2.resize(img, (int(img.shape[1]*h/img.shape[0]), h))
                vis = np.hstack([rh(frame_bgr), rh(depth_vis)])

                # FPS
                frames += 1
                dt = time.time() - t0
                if dt >= 1.0:
                    fps = frames / dt
                    t0, frames = time.time(), 0
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

                # Show centre depth
                h2, w2 = frame_bgr.shape[:2]
                u, v = w2 // 2, h2 // 2
                Z = float(depth_m[v, u])
                if np.isfinite(Z) and Z > 0:
                    cv2.putText(vis, f"Centre depth: {Z:.2f} m", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("ZED EO + Depth", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
            else:
                time.sleep(0.002)
    finally:
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()