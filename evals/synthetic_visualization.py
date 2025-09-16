#!/usr/bin/env python3
"""
Synthetic Tracking Visualization Tool

This script visualizes the results from synthetic tracking benchmarks.
It shows ground truth vs. tracked positions and error metrics.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import cv2
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import glob


def find_latest_experiment():
    """Find the most recent experiment folder in the logs directory"""
    experiment_dirs = [d for d in glob.glob(os.path.join('logs', '*')) 
                      if os.path.isdir(d)]
    
    if not experiment_dirs:
        return None
    
    # Sort by modification time (newest first)
    experiment_dirs.sort(key=os.path.getmtime, reverse=True)
    
    # Check for evaluation_data.json in the most recent experiment folder
    for exp_dir in experiment_dirs:
        eval_file = os.path.join(exp_dir, 'evaluation_data.json')
        if os.path.exists(eval_file):
            return exp_dir
    
    return None


def load_evaluation_data(experiment_path):
    """Load evaluation data from an experiment folder"""
    eval_path = os.path.join(experiment_path, 'evaluation_data.json')
    
    if not os.path.exists(eval_path):
        print(f"No evaluation data found at {eval_path}")
        return None
    
    try:
        with open(eval_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to tuples for ground_truth and tracked_positions
        for key in ['ground_truth', 'tracked_positions', 'servo_positions']:
            if key in data:
                # Each entry is [time, [x, y]] or [time, null]
                for i, entry in enumerate(data[key]):
                    if entry[1] is not None:
                        data[key][i][1] = tuple(entry[1])
        
        return data
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return None


def load_log_data(experiment_path):
    """Load tracking log data from an experiment folder"""
    log_path = os.path.join(experiment_path, 'tracking_data.log')
    
    if not os.path.exists(log_path):
        print(f"No log data found at {log_path}")
        return None
    
    try:
        df = pd.read_csv(log_path)
        return df
    except Exception as e:
        print(f"Error loading log data: {e}")
        return None


def calculate_metrics(eval_data):
    """Calculate tracking performance metrics"""
    if not eval_data:
        return None
        
    ground_truth = eval_data['ground_truth']
    tracked = eval_data['tracked_positions']
    
    # Convert to numpy arrays for easier processing
    gt_times = np.array([entry[0] for entry in ground_truth])
    gt_positions = np.array([entry[1] for entry in ground_truth])
    
    tr_times = np.array([entry[0] for entry in tracked])
    tr_positions = []
    for entry in tracked:
        if entry[1] is not None:
            tr_positions.append(entry[1])
        else:
            tr_positions.append((np.nan, np.nan))
    tr_positions = np.array(tr_positions)
    
    # Calculate pixel errors for each frame
    errors = []
    total_frames = len(ground_truth)
    tracked_frames = 0
    
    for i in range(len(ground_truth)):
        if i < len(tracked) and tracked[i][1] is not None:
            # Calculate error in pixels
            gt_pos = ground_truth[i][1]
            tr_pos = tracked[i][1]
            
            error_x = gt_pos[0] - tr_pos[0]
            error_y = gt_pos[1] - tr_pos[1]
            error_distance = np.sqrt(error_x ** 2 + error_y ** 2)
            
            errors.append(error_distance)
            tracked_frames += 1
    
    if not errors:
        print("No tracking matches found for evaluation")
        return None
        
    # Calculate metrics
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    tracking_rate = tracked_frames / total_frames
    
    # Calculate average servo lag (time between ground truth movement and servo response)
    # This is more complex and would require aligning the servo movements with ground truth
    
    metrics = {
        'total_frames': total_frames,
        'tracked_frames': tracked_frames,
        'tracking_rate': tracking_rate,
        'average_error_pixels': avg_error,
        'median_error_pixels': median_error,
        'max_error_pixels': max_error,
        'min_error_pixels': min_error,
        'std_error_pixels': std_error
    }
    
    return metrics


def plot_tracking_comparison(eval_data, metrics, output_path=None):
    """Plot comparison of ground truth vs tracked positions"""
    if not eval_data or not metrics:
        return
    
    # Extract data
    ground_truth = eval_data['ground_truth']
    tracked = eval_data['tracked_positions']
    servos = eval_data['servo_positions']
    config = eval_data['config']
    
    # Get movement pattern info
    pattern = config['simulation']['movement_pattern']
    pattern_params = config['simulation']['pattern_params']
    pattern_description = f"{pattern.capitalize()} pattern"
    if pattern == 'circular':
        radius = pattern_params.get('radius', 0)
        pattern_description += f" (radius: {radius}px)"
    elif pattern == 'linear':
        direction = pattern_params.get('direction', 'left_to_right')
        pattern_description += f" ({direction})"
    
    # Extract positions and times
    gt_times = np.array([entry[0] for entry in ground_truth])
    gt_x = np.array([entry[1][0] if entry[1] is not None else np.nan for entry in ground_truth])
    gt_y = np.array([entry[1][1] if entry[1] is not None else np.nan for entry in ground_truth])
    
    tr_times = np.array([entry[0] for entry in tracked])
    tr_x = np.array([entry[1][0] if entry[1] is not None else np.nan for entry in tracked])
    tr_y = np.array([entry[1][1] if entry[1] is not None else np.nan for entry in tracked])
    
    # Remove NaN values for path plotting
    gt_x_clean = gt_x[~np.isnan(gt_x)]
    gt_y_clean = gt_y[~np.isnan(gt_y)]
    tr_x_clean = tr_x[~np.isnan(tr_x)]
    tr_y_clean = tr_y[~np.isnan(tr_y)]
    
    # Extract servo positions
    servo_times = np.array([entry[0] for entry in servos])
    servo_pan = np.array([entry[1][0] for entry in servos])
    servo_tilt = np.array([entry[1][1] for entry in servos])
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot 1: Movement paths (2D plot)
    ax_paths = fig.add_subplot(gs[0, :])
    ax_paths.plot(gt_x_clean, gt_y_clean, 'b-', linewidth=2, label='Ground Truth')
    ax_paths.plot(tr_x_clean, tr_y_clean, 'r--', linewidth=2, label='Tracked Path')
    ax_paths.set_title('Ground Truth vs Tracked Path')
    ax_paths.set_xlabel('X Position (pixels)')
    ax_paths.set_ylabel('Y Position (pixels)')
    ax_paths.legend()
    ax_paths.grid(True)
    ax_paths.axis('equal')  # Equal aspect ratio
    
    # Plot 2: X position over time
    ax_x = fig.add_subplot(gs[1, 0])
    ax_x.plot(gt_times - gt_times[0], gt_x, 'b-', label='Ground Truth')
    ax_x.plot(tr_times - tr_times[0], tr_x, 'r--', label='Tracked')
    ax_x.set_title('X Position Over Time')
    ax_x.set_xlabel('Time (seconds)')
    ax_x.set_ylabel('X Position (pixels)')
    ax_x.legend()
    ax_x.grid(True)
    
    # Plot 3: Y position over time
    ax_y = fig.add_subplot(gs[1, 1])
    ax_y.plot(gt_times - gt_times[0], gt_y, 'b-', label='Ground Truth')
    ax_y.plot(tr_times - tr_times[0], tr_y, 'r--', label='Tracked')
    ax_y.set_title('Y Position Over Time')
    ax_y.set_xlabel('Time (seconds)')
    ax_y.set_ylabel('Y Position (pixels)')
    ax_y.legend()
    ax_y.grid(True)
    
    # Plot 4: Tracking error over time
    ax_error = fig.add_subplot(gs[2, 0])
    errors = []
    error_times = []
    
    for i in range(min(len(ground_truth), len(tracked))):
        if tracked[i][1] is not None:
            gt_pos = ground_truth[i][1]
            tr_pos = tracked[i][1]
            
            error_x = gt_pos[0] - tr_pos[0]
            error_y = gt_pos[1] - tr_pos[1]
            error_distance = np.sqrt(error_x ** 2 + error_y ** 2)
            
            errors.append(error_distance)
            error_times.append(tracked[i][0])
    
    if errors:
        ax_error.plot(np.array(error_times) - gt_times[0], errors, 'g-')
        ax_error.set_title('Tracking Error Over Time')
        ax_error.set_xlabel('Time (seconds)')
        ax_error.set_ylabel('Error Distance (pixels)')
        ax_error.grid(True)
    
    # Plot 5: Servo positions over time
    ax_servo = fig.add_subplot(gs[2, 1])
    ax_servo.plot(servo_times - servo_times[0], servo_pan, 'b-', label='Pan')
    ax_servo.plot(servo_times - servo_times[0], servo_tilt, 'r-', label='Tilt')
    ax_servo.set_title('Servo Positions Over Time')
    ax_servo.set_xlabel('Time (seconds)')
    ax_servo.set_ylabel('Angle (degrees)')
    ax_servo.legend()
    ax_servo.grid(True)
    
    # Add experiment information
    title = f"Tracking Performance - {pattern_description}"
    fig.suptitle(title, fontsize=16)
    
    # Add metrics text box
    metrics_text = (
        f"Tracking rate: {metrics['tracking_rate']:.1%}\n"
        f"Average error: {metrics['average_error_pixels']:.1f} px\n"
        f"Median error: {metrics['median_error_pixels']:.1f} px\n"
        f"Max error: {metrics['max_error_pixels']:.1f} px\n"
        f"Min error: {metrics['min_error_pixels']:.1f} px\n"
        f"Std deviation: {metrics['std_error_pixels']:.1f} px"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax_paths.text(0.05, 0.95, metrics_text, transform=ax_paths.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    
    # Save the figure if an output path is provided
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Tracking comparison saved to: {output_path}")
    
    # Show the figure
    plt.show()


def create_tracking_animation(eval_data, output_path=None):
    """Create an animation of the ground truth vs. tracked positions"""
    if not eval_data:
        return
    
    # Extract data
    ground_truth = eval_data['ground_truth']
    tracked = eval_data['tracked_positions']
    config = eval_data['config']
    
    # Determine frame size from config
    frame_width = config['simulation']['frame_width']
    frame_height = config['simulation']['frame_height']
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, frame_width)
    ax.set_ylim(0, frame_height)
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Tracking Animation: Ground Truth vs. Tracked Position')
    ax.grid(True)
    
    # Plot initial empty lines
    gt_line, = ax.plot([], [], 'b-', linewidth=2, label='Ground Truth Path')
    tr_line, = ax.plot([], [], 'r--', linewidth=2, label='Tracked Path')
    
    # Current position markers
    gt_point, = ax.plot([], [], 'bo', markersize=8, label='Ground Truth')
    tr_point, = ax.plot([], [], 'ro', markersize=8, label='Tracked Position')
    
    # Dead zone (from config)
    dead_zone = config['tracking'].get('dead_zone', 50)
    frame_center = (frame_width // 2, frame_height // 2)
    center_circle = plt.Circle(frame_center, dead_zone, color='blue', fill=False, linestyle='--')
    ax.add_patch(center_circle)
    
    # Mark frame center
    ax.plot(frame_center[0], frame_center[1], 'bx', markersize=10)
    
    # Legend
    ax.legend()
    
    # Data containers
    gt_x_data, gt_y_data = [], []
    tr_x_data, tr_y_data = [], []
    
    def init():
        gt_line.set_data([], [])
        tr_line.set_data([], [])
        gt_point.set_data([], [])
        tr_point.set_data([], [])
        return gt_line, tr_line, gt_point, tr_point
    
    def animate(i):
        # Update ground truth path
        if ground_truth[i][1] is not None:
            gt_x, gt_y = ground_truth[i][1]
            gt_x_data.append(gt_x)
            gt_y_data.append(gt_y)
            gt_line.set_data(gt_x_data, gt_y_data)
            gt_point.set_data([gt_x], [gt_y])
        
        # Update tracked position path
        if i < len(tracked) and tracked[i][1] is not None:
            tr_x, tr_y = tracked[i][1]
            tr_x_data.append(tr_x)
            tr_y_data.append(tr_y)
            tr_line.set_data(tr_x_data, tr_y_data)
            tr_point.set_data([tr_x], [tr_y])
        else:
            # No tracking data for this frame
            tr_point.set_data([], [])
        
        # Update title with frame information
        ax.set_title(f'Tracking Animation: Frame {i+1}/{len(ground_truth)}')
        
        return gt_line, tr_line, gt_point, tr_point
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(ground_truth), interval=50, blit=True
    )
    
    # Save animation if output path provided
    if output_path:
        anim.save(output_path, writer='pillow', fps=20)
        print(f"Animation saved to: {output_path}")
    
    plt.tight_layout()
    plt.show()


def create_tracking_comparison_video(eval_data, output_path=None):
    """Create a video visualization of the tracking performance"""
    if not eval_data:
        return
    
    # Extract data
    ground_truth = eval_data['ground_truth']
    tracked = eval_data['tracked_positions']
    servos = eval_data['servo_positions']
    config = eval_data['config']
    
    # Determine frame size from config
    frame_width = config['simulation']['frame_width']
    frame_height = config['simulation']['frame_height']
    
    # Create video output
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20, (frame_width, frame_height))
    else:
        out = None
    
    # Data containers for paths
    gt_points = []
    tr_points = []
    
    # Frame center and dead zone
    frame_center = (frame_width // 2, frame_height // 2)
    dead_zone = config['tracking'].get('dead_zone', 50)
    
    # Create frames
    for i in range(len(ground_truth)):
        # Create a blank frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Draw grid lines
        for x in range(0, frame_width, 100):
            cv2.line(frame, (x, 0), (x, frame_height), (20, 20, 20), 1)
        for y in range(0, frame_height, 100):
            cv2.line(frame, (0, y), (frame_width, y), (20, 20, 20), 1)
        
        # Draw frame center and dead zone
        cv2.circle(frame, frame_center, 5, (255, 255, 255), -1)
        cv2.circle(frame, frame_center, dead_zone, (100, 100, 255), 1)
        
        # Draw previous ground truth path
        if ground_truth[i][1] is not None:
            gt_points.append(tuple(map(int, ground_truth[i][1])))
            
            # Draw the ground truth path with a blue line
            if len(gt_points) > 1:
                for j in range(1, len(gt_points)):
                    cv2.line(frame, gt_points[j-1], gt_points[j], (255, 0, 0), 2)
            
            # Draw current ground truth position
            cv2.circle(frame, gt_points[-1], 10, (255, 0, 0), -1)
        
        # Draw tracked path
        if i < len(tracked) and tracked[i][1] is not None:
            tr_points.append(tuple(map(int, tracked[i][1])))
            
            # Draw the tracked path with a red line
            if len(tr_points) > 1:
                for j in range(1, len(tr_points)):
                    cv2.line(frame, tr_points[j-1], tr_points[j], (0, 0, 255), 2)
            
            # Draw current tracked position
            cv2.circle(frame, tr_points[-1], 8, (0, 0, 255), -1)
        
        # Calculate and display error
        if i < len(tracked) and tracked[i][1] is not None and ground_truth[i][1] is not None:
            gt_pos = ground_truth[i][1]
            tr_pos = tracked[i][1]
            
            error_x = gt_pos[0] - tr_pos[0]
            error_y = gt_pos[1] - tr_pos[1]
            error_distance = np.sqrt(error_x ** 2 + error_y ** 2)
            
            # Draw line between ground truth and tracked position
            cv2.line(frame, tuple(map(int, gt_pos)), tuple(map(int, tr_pos)), (0, 255, 0), 1)
            
            # Display error text
            cv2.putText(frame, f"Error: {error_distance:.1f} px", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame number and time
        time_elapsed = ground_truth[i][0] - ground_truth[0][0]
        cv2.putText(frame, f"Frame: {i+1}/{len(ground_truth)}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {time_elapsed:.2f}s", (frame_width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display movement pattern
        pattern = config['simulation']['movement_pattern']
        cv2.putText(frame, f"Pattern: {pattern}", (20, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display servo positions
        if i < len(servos):
            pan, tilt = servos[i][1]
            cv2.putText(frame, f"Pan: {pan:.1f}°  Tilt: {tilt:.1f}°", (frame_width - 300, frame_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Add legend
        cv2.circle(frame, (20, frame_height - 60), 8, (255, 0, 0), -1)
        cv2.putText(frame, "Ground Truth", (40, frame_height - 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.circle(frame, (20, frame_height - 90), 8, (0, 0, 255), -1)
        cv2.putText(frame, "Tracked Position", (40, frame_height - 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Write frame to video
        if out:
            out.write(frame)
        
        # Display frame
        cv2.imshow("Tracking Visualization", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
    
    # Clean up
    if out:
        out.release()
        print(f"Video saved to: {output_path}")
    
    cv2.destroyAllWindows()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize Synthetic Tracking Benchmark Results')
    parser.add_argument('--experiment', '-e', type=str, default=None,
                       help='Experiment folder name')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path for plots')
    parser.add_argument('--video', '-v', action='store_true',
                       help='Create video visualization')
    parser.add_argument('--animation', '-a', action='store_true',
                       help='Create animation visualization')
    
    args = parser.parse_args()
    
    # Find experiment folder
    experiment_path = None
    if args.experiment:
        experiment_path = os.path.join('logs', args.experiment)
        if not os.path.isdir(experiment_path):
            print(f"Experiment folder '{args.experiment}' not found")
            return
    else:
        # Use latest experiment folder
        experiment_path = find_latest_experiment()
        if not experiment_path:
            print("No experiment folders found with evaluation data")
            return
    
    # Load evaluation data
    eval_data = load_evaluation_data(experiment_path)
    if not eval_data:
        return
    
    # Calculate metrics
    metrics = calculate_metrics(eval_data)
    if not metrics:
        return
    
    print("Tracking Performance Metrics:")
    print(f"- Total frames: {metrics['total_frames']}")
    print(f"- Tracked frames: {metrics['tracked_frames']} ({metrics['tracking_rate']:.1%})")
    print(f"- Average error: {metrics['average_error_pixels']:.2f} pixels")
    print(f"- Median error: {metrics['median_error_pixels']:.2f} pixels")
    print(f"- Max error: {metrics['max_error_pixels']:.2f} pixels")
    print(f"- Min error: {metrics['min_error_pixels']:.2f} pixels")
    print(f"- Error standard deviation: {metrics['std_error_pixels']:.2f} pixels")
    
    # Determine output paths
    experiment_name = os.path.basename(experiment_path)
    output_dir = experiment_path
    
    if args.output:
        output_plot = args.output
    else:
        output_plot = os.path.join(output_dir, 'tracking_comparison.png')
    
    # Create plots
    plot_tracking_comparison(eval_data, metrics, output_plot)
    
    # Create animation if requested
    if args.animation:
        animation_path = os.path.join(output_dir, 'tracking_animation.gif')
        create_tracking_animation(eval_data, animation_path)
    
    # Create video if requested
    if args.video:
        video_path = os.path.join(output_dir, 'tracking_video.avi')
        create_tracking_comparison_video(eval_data, video_path)


if __name__ == "__main__":
    main()