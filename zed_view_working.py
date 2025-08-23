#!/usr/bin/env python3
import numpy as np
import cv2
import pyzed.sl as sl
import time

def colourise_depth(depth_m: np.ndarray, dmin=0.3, dmax=5.0) -> np.ndarray:
    """Map 32F depth (m) to a colour image. Invalids -> black."""
    # Make a copy and ensure proper data type
    d = np.array(depth_m, dtype=np.float32, copy=True)
    
    # Handle invalid values
    invalid = ~np.isfinite(d) | (d <= 0)
    d[invalid] = dmax
    d = np.clip(d, dmin, dmax)
    
    # Normalize to 0-255 range
    d_norm = ((d - dmin) / (dmax - dmin) * 255.0)
    d_norm = d_norm.astype(np.uint8)
    
    # Invert for colormap (nearer = warmer) and ensure contiguous array
    d_inverted = 255 - d_norm
    d_inverted = np.ascontiguousarray(d_inverted, dtype=np.uint8)
    
    # Debug print to check array properties
    print(f"d_inverted type: {type(d_inverted)}, dtype: {d_inverted.dtype}, shape: {d_inverted.shape}")
    print(f"Is contiguous: {d_inverted.flags['C_CONTIGUOUS']}")
    print(f"Min/Max values: {d_inverted.min()}/{d_inverted.max()}")
    
    # Apply colormap
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    try:
        depth_color = cv2.applyColorMap(d_inverted, cmap)
        depth_color[invalid] = (0, 0, 0)
        return depth_color
    except Exception as e:
        print(f"OpenCV applyColorMap error: {e}")
        # Fallback: create a simple grayscale version
        depth_gray = cv2.cvtColor(d_inverted, cv2.COLOR_GRAY2BGR)
        depth_gray[invalid] = (0, 0, 0)
        return depth_gray

def main():
    zed = sl.Camera()
    init_params = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.HD720,      # good balance
        depth_mode=sl.DEPTH_MODE.PERFORMANCE,       # fastest
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
                zed.retrieve_image(left, sl.VIEW.LEFT)          # BGRA, uint8
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)   # 32F, metres

                frame_bgra = left.get_data()
                frame_bgr = frame_bgra[..., :3].copy()
                depth_m = depth.get_data()
                
                # Only debug on first frame
                if frames == 0:
                    depth_vis = colourise_depth(depth_m, dmin=0.3, dmax=5.0)
                else:
                    # Simplified version after debugging
                    d = np.array(depth_m, dtype=np.float32, copy=True)
                    invalid = ~np.isfinite(d) | (d <= 0)
                    d[invalid] = 5.0
                    d = np.clip(d, 0.3, 5.0)
                    d_norm = ((d - 0.3) / (5.0 - 0.3) * 255.0).astype(np.uint8)
                    d_inverted = np.ascontiguousarray(255 - d_norm, dtype=np.uint8)
                    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
                    depth_vis = cv2.applyColorMap(d_inverted, cmap)
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

                # Optional: show centre depth in metres
                h2, w2 = frame_bgr.shape[:2]
                u, v = w2 // 2, h2 // 2
                Z = float(depth_m[v, u])
                if np.isfinite(Z) and Z > 0:
                    cv2.putText(vis, f"Centre depth: {Z:.2f} m", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("ZED EO + Depth", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # ESC or q
                    break
            else:
                # Brief backoff if grab fails
                time.sleep(0.002)
    finally:
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()