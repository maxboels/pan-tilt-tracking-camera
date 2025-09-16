#!/usr/bin/env python3
"""
Interactive visualization tool for pan-tilt tracking performance
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from pathlib import Path
from matplotlib.patches import Circle, Rectangle
import csv
import io
import datetime  # Add this import to fix the datetime error

# Import our log parsing function
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from evals.analyze_tracking_logs import parse_log_file, find_latest_experiment
except ImportError:
    # Define parsing function here if import fails
    def find_latest_experiment():
        """Find the most recent experiment directory"""
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        experiment_dirs = glob.glob(os.path.join(base_dir, "experiment_*"))
        
        if not experiment_dirs:
            print("No experiment directories found")
            return None
        
        # Sort by modification time (most recent first)
        experiment_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return os.path.basename(experiment_dirs[0])
    
    # Define a simple CSV parser if import fails
    def parse_log_file(tracking_log):
        """Simple log file parser"""
        data = []
        with open(tracking_log, 'r') as f:
            # Read header row
            headers = f.readline().strip().split(',')
            
            # Read data rows
            for line in f:
                values = line.strip().split(',')
                if len(values) == len(headers):
                    entry = {}
                    for i, header in enumerate(headers):
                        try:
                            if values[i].lower() == 'true':
                                entry[header] = True
                            elif values[i].lower() == 'false':
                                entry[header] = False
                            elif '.' in values[i]:
                                entry[header] = float(values[i])
                            else:
                                try:
                                    entry[header] = int(values[i])
                                except ValueError:
                                    entry[header] = values[i]
                        except:
                            entry[header] = values[i]
                    data.append(entry)
        return pd.DataFrame(data)

def load_tracking_data(experiment_name=None, log_file=None):
    """
    Load tracking data from experiment logs
    
    Args:
        experiment_name: Name of experiment directory
        log_file: Direct path to log file
        
    Returns:
        df: Pandas DataFrame with tracking data
        config: Configuration dictionary
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Determine the log file path
    if log_file:
        tracking_log = log_file
    elif experiment_name:
        tracking_log = os.path.join(base_dir, "logs", experiment_name, "tracking_data.log")
    else:
        # Find the most recent experiment
        experiment_name = find_latest_experiment()
        if not experiment_name:
            print("No experiments found to analyze")
            return None, None
        tracking_log = os.path.join(base_dir, "logs", experiment_name, "tracking_data.log")
    
    print(f"Loading tracking data from: {tracking_log}")
    
    # Check if log file exists
    if not os.path.exists(tracking_log):
        print(f"Error: Log file not found: {tracking_log}")
        return None, None
    
    # Load configuration if available
    config_file = os.path.join(os.path.dirname(tracking_log), "system_config.json")
    config = None
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    # Parse log file
    df = parse_log_file(tracking_log)
    if df is None or df.empty:
        print("Failed to parse log file or no valid data found")
        return None, None
    
    # Process the data for visualization
    process_data_for_visualization(df)
    
    return df, config

def process_data_for_visualization(df):
    """Process dataframe to prepare for visualization"""
    # Find the relevant columns
    target_x_col = None
    target_y_col = None
    center_x_col = None
    center_y_col = None
    
    # Check for different possible column names for target position
    for col_name in ['target_x', 'smoothed_target_x', 'target_center_x']:
        if col_name in df.columns:
            target_x_col = col_name
            break
    
    for col_name in ['target_y', 'smoothed_target_y', 'target_center_y']:
        if col_name in df.columns:
            target_y_col = col_name
            break
    
    # Check for different possible column names for camera center
    for col_name in ['camera_center_x', 'frame_center_x']:
        if col_name in df.columns:
            center_x_col = col_name
            break
    
    for col_name in ['camera_center_y', 'frame_center_y']:
        if col_name in df.columns:
            center_y_col = col_name
            break
    
    # Calculate tracking errors if needed columns are present
    if target_x_col and target_y_col and center_x_col and center_y_col:
        df['tracking_error_x'] = df[target_x_col] - df[center_x_col]
        df['tracking_error_y'] = df[target_y_col] - df[center_y_col]
        df['tracking_error_dist'] = np.sqrt(df['tracking_error_x']**2 + df['tracking_error_y']**2)
        
        # Add quadrant info
        df['quadrant'] = df.apply(
            lambda row: determine_quadrant(row['tracking_error_x'], row['tracking_error_y']),
            axis=1
        )
    else:
        print("Warning: Required columns for tracking error calculation not found")

def determine_quadrant(x_error, y_error):
    """Determine which quadrant of the frame the target is in"""
    if x_error > 0 and y_error < 0:
        return "TOP-RIGHT"
    elif x_error < 0 and y_error < 0:
        return "TOP-LEFT"
    elif x_error > 0 and y_error > 0:
        return "BOTTOM-RIGHT"
    elif x_error < 0 and y_error > 0:
        return "BOTTOM-LEFT"
    else:
        return "CENTER"

def interactive_visualization(df, config=None):
    """Create interactive visualization of tracking data"""
    if df is None or df.empty:
        print("No valid data to visualize")
        return
    
    # Check if we have the necessary columns
    if not all(col in df.columns for col in 
              ['tracking_error_x', 'tracking_error_y', 'tracking_error_dist']):
        print("Required tracking error columns not found in data")
        return
    
    # Filter for frames with detections
    if 'target_detected' in df.columns:
        plot_df = df[df.target_detected == True].copy()
    else:
        plot_df = df.copy()
    
    if len(plot_df) == 0:
        print("No frames with detections to visualize")
        return
    
    # Get camera resolution
    frame_width = 1920
    frame_height = 1080
    
    if config and 'app_config' in config and 'camera' in config['app_config']:
        camera_config = config['app_config']['camera']
        if 'resolution' in camera_config:
            frame_width = camera_config['resolution'][0]
            frame_height = camera_config['resolution'][1]
    
    # Calculate aspect ratio for the scatter plot
    aspect_ratio = frame_height / frame_width
    
    # Create interactive figure
    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title('Pan-Tilt Tracking Performance Visualization')
    
    # Layout for plots and sliders
    gs = fig.add_gridspec(3, 3)
    
    # Plot 1: Target position relative to center (scatter)
    ax_scatter = fig.add_subplot(gs[0:2, 0:2])
    scatter = ax_scatter.scatter(plot_df.tracking_error_x, plot_df.tracking_error_y, 
                                c=plot_df.index, cmap='viridis', alpha=0.7)
    
    # Draw crosshairs
    ax_scatter.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_scatter.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    # Get dead zone size from config
    dead_zone_size = 20  # Default
    if config and 'app_config' in config and 'pantilt' in config['app_config']:
        if 'dead_zone' in config['app_config']['pantilt']:
            dead_zone_size = config['app_config']['pantilt']['dead_zone']
    
    # Draw reference circles
    for radius in [dead_zone_size, 50, 100, 200]:
        circle = Circle((0, 0), radius, color='r', fill=False, alpha=0.3)
        ax_scatter.add_patch(circle)
        ax_scatter.text(radius*0.7, radius*0.3, f"{radius}px", color='r', alpha=0.7)
    
    # Set axis limits with margin
    max_error = max(
        abs(plot_df.tracking_error_x.min()), 
        abs(plot_df.tracking_error_x.max()),
        abs(plot_df.tracking_error_y.min()), 
        abs(plot_df.tracking_error_y.max())
    )
    margin = max(100, max_error * 1.1)
    
    ax_scatter.set_xlim(-margin, margin)
    ax_scatter.set_ylim(-margin * aspect_ratio, margin * aspect_ratio)
    ax_scatter.set_aspect(aspect_ratio)
    ax_scatter.set_title('Target Position Relative to Camera Center')
    ax_scatter.set_xlabel('X Error (pixels)')
    ax_scatter.set_ylabel('Y Error (pixels)')
    ax_scatter.grid(True)
    
    # Plot 2: Error over time
    ax_error = fig.add_subplot(gs[0, 2])
    ax_error.plot(plot_df.index, plot_df.tracking_error_dist)
    ax_error.axhline(y=dead_zone_size, color='g', linestyle='--', label='Dead zone')
    ax_error.set_title('Total Error Distance Over Time')
    ax_error.set_xlabel('Frame')
    ax_error.set_ylabel('Error (pixels)')
    ax_error.grid(True)
    ax_error.legend()
    
    # Plot 3: X error over time
    ax_x_error = fig.add_subplot(gs[1, 2])
    ax_x_error.plot(plot_df.index, plot_df.tracking_error_x, 'b-')
    ax_x_error.axhline(y=0, color='r', linestyle='--')
    ax_x_error.set_title('X Error Over Time')
    ax_x_error.set_xlabel('Frame')
    ax_x_error.set_ylabel('X Error (pixels)')
    ax_x_error.grid(True)
    
    # Plot 4: Y error over time
    ax_y_error = fig.add_subplot(gs[2, 0])
    ax_y_error.plot(plot_df.index, plot_df.tracking_error_y, 'g-')
    ax_y_error.axhline(y=0, color='r', linestyle='--')
    ax_y_error.set_title('Y Error Over Time')
    ax_y_error.set_xlabel('Frame')
    ax_y_error.set_ylabel('Y Error (pixels)')
    ax_y_error.grid(True)
    
    # Plot 5: Quadrant distribution
    ax_quadrant = fig.add_subplot(gs[2, 1])
    quadrant_counts = plot_df['quadrant'].value_counts()
    ax_quadrant.bar(quadrant_counts.index, quadrant_counts.values)
    ax_quadrant.set_title('Target Position Distribution')
    ax_quadrant.set_xlabel('Quadrant')
    ax_quadrant.set_ylabel('Count')
    plt.setp(ax_quadrant.get_xticklabels(), rotation=45, ha='right')
    
    # Add a slider for frame selection
    ax_slider = fig.add_subplot(gs[2, 2])
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=plot_df.index.min(),
        valmax=plot_df.index.max(),
        valinit=plot_df.index.min()
    )
    
    # Add a vertical line to track current frame position
    line_error = ax_error.axvline(x=plot_df.index.min(), color='k', alpha=0.5)
    line_x_error = ax_x_error.axvline(x=plot_df.index.min(), color='k', alpha=0.5)
    line_y_error = ax_y_error.axvline(x=plot_df.index.min(), color='k', alpha=0.5)
    
    # Current position indicator in scatter plot
    highlight_point = ax_scatter.scatter([], [], s=150, facecolors='none', edgecolors='black', linewidths=2)
    
    # Frame info text
    frame_info = ax_scatter.text(0.05, 0.95, "", transform=ax_scatter.transAxes, 
                             va='top', ha='left', bbox=dict(boxstyle="round", fc="w", alpha=0.8))
    
    # Add stats to the figure
    avg_error = plot_df.tracking_error_dist.mean()
    in_deadzone = (plot_df.tracking_error_dist <= dead_zone_size).sum() / len(plot_df) * 100
    
    stats_text = (
        f"Total Frames: {len(df)}\n"
        f"Frames with Detection: {len(plot_df)}\n"
        f"Avg. Error: {avg_error:.2f} px\n"
        f"Max Error: {plot_df.tracking_error_dist.max():.2f} px\n"
        f"In Dead Zone: {in_deadzone:.1f}%\n"
        f"Camera: {frame_width}x{frame_height}"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, 
            bbox=dict(boxstyle="round", fc="w", alpha=0.8))
    
    # Update function for slider
    def update(val):
        # Find the closest frame index
        frame_idx = min(plot_df.index, key=lambda x: abs(x - val))
        
        # Update vertical lines on time plots
        line_error.set_xdata(frame_idx)
        line_x_error.set_xdata(frame_idx)
        line_y_error.set_xdata(frame_idx)
        
        # Update highlighted point on scatter plot
        row = plot_df.loc[frame_idx]
        highlight_point.set_offsets([row.tracking_error_x, row.tracking_error_y])
        
        # Update frame info text
        frame_info_str = (
            f"Frame: {frame_idx}\n"
            f"Error: {row.tracking_error_dist:.1f} px\n"
            f"X Error: {row.tracking_error_x:.1f} px\n"
            f"Y Error: {row.tracking_error_y:.1f} px\n"
            f"Quadrant: {row.quadrant}"
        )
        frame_info.set_text(frame_info_str)
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Add play/pause animation
    ax_play = plt.axes([0.8, 0.025, 0.1, 0.04])
    button_play = Button(ax_play, 'Play/Pause')
    
    animation_running = [False]  # Use a list to store state so it can be modified in closure
    anim = None
    
    def animate(frame):
        slider.set_val(frame)
        return highlight_point, line_error, line_x_error, line_y_error
    
    def toggle_animation(event):
        nonlocal anim
        if animation_running[0]:
            anim.event_source.stop()
            animation_running[0] = False
        else:
            if anim is None:
                frames = plot_df.index.tolist()
                anim = animation.FuncAnimation(
                    fig, animate, frames=frames, 
                    interval=100, blit=False, repeat=True
                )
            else:
                anim.event_source.start()
            animation_running[0] = True
    
    button_play.on_clicked(toggle_animation)
    
    plt.tight_layout()
    update(plot_df.index.min())  # Initialize with first frame
    plt.show()

def save_animation(df, output_path=None, config=None):
    """Create and save an animation of the tracking performance"""
    if df is None or df.empty:
        print("No valid data for animation")
        return
    
    # Check if we have the necessary columns
    if not all(col in df.columns for col in 
              ['tracking_error_x', 'tracking_error_y', 'tracking_error_dist']):
        print("Required tracking error columns not found in data")
        return
    
    # Filter for frames with detections
    if 'target_detected' in df.columns:
        plot_df = df[df.target_detected == True].copy()
    else:
        plot_df = df.copy()
    
    if len(plot_df) == 0:
        print("No frames with detections for animation")
        return
    
    # Default output path
    if output_path is None:
        # Determine path based on experiment folder
        try:
            # Try to extract experiment name from the dataframe source
            if hasattr(df, 'experiment_name') and df.experiment_name:
                exp_name = df.experiment_name
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                output_path = os.path.join(base_dir, "logs", exp_name, "analysis", "tracking_animation.mp4")
            else:
                output_path = 'tracking_animation.mp4'
        except:
            output_path = 'tracking_animation.mp4'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get camera resolution
    frame_width = 1920
    frame_height = 1080
    
    if config and 'app_config' in config and 'camera' in config['app_config']:
        camera_config = config['app_config']['camera']
        if 'resolution' in camera_config:
            frame_width = camera_config['resolution'][0]
            frame_height = camera_config['resolution'][1]
    
    aspect_ratio = frame_height / frame_width
    
    # Get dead zone size from config
    dead_zone_size = 20  # Default
    if config and 'app_config' in config and 'pantilt' in config['app_config']:
        if 'dead_zone' in config['app_config']['pantilt']:
            dead_zone_size = config['app_config']['pantilt']['dead_zone']
    elif config and 'pantilt' in config:
        if 'dead_zone' in config['pantilt']:
            dead_zone_size = config['pantilt']['dead_zone']
    
    # Set up the figure
    fig, (ax_scatter, ax_error) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Calculate axis limits
    max_error = max(
        abs(plot_df.tracking_error_x.min()), 
        abs(plot_df.tracking_error_x.max()),
        abs(plot_df.tracking_error_y.min()), 
        abs(plot_df.tracking_error_y.max())
    )
    margin = max(100, max_error * 1.1)
    
    # Set up scatter plot
    scatter = ax_scatter.scatter([], [], c='blue', alpha=0.7)
    highlight_point = ax_scatter.scatter([], [], s=150, facecolors='none', edgecolors='black', linewidths=2)
    
    # Draw crosshairs and circles
    ax_scatter.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_scatter.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    for radius in [dead_zone_size, 50, 100, 200]:
        circle = Circle((0, 0), radius, color='r', fill=False, alpha=0.3)
        ax_scatter.add_patch(circle)
        # Add label showing radius
        ax_scatter.text(radius*0.7, radius*0.3, f"{radius}px", color='r', alpha=0.7)
    
    ax_scatter.set_xlim(-margin, margin)
    ax_scatter.set_ylim(-margin * aspect_ratio, margin * aspect_ratio)
    ax_scatter.set_title('Target Position Relative to Camera Center')
    ax_scatter.set_xlabel('X Error (pixels)')
    ax_scatter.set_ylabel('Y Error (pixels)')
    ax_scatter.grid(True)
    
    # Add explanatory notes
    exp_text = "CAMERA CENTER\n\n"
    exp_text += "Target Position Legend:\n"
    exp_text += "• TOP-LEFT: Camera should turn LEFT and UP\n"
    exp_text += "• TOP-RIGHT: Camera should turn RIGHT and UP\n"
    exp_text += "• BOTTOM-LEFT: Camera should turn LEFT and DOWN\n"
    exp_text += "• BOTTOM-RIGHT: Camera should turn RIGHT and DOWN\n"
    
    ax_scatter.text(0, -margin*0.9*aspect_ratio, exp_text,
                    ha='center', va='bottom', 
                    bbox=dict(boxstyle="round", fc="whitesmoke", alpha=0.8),
                    fontsize=9)
    
    # Set up error plot
    line, = ax_error.plot([], [], lw=2)
    ax_error.axhline(y=dead_zone_size, color='g', linestyle='--', label=f'Dead zone ({dead_zone_size}px)')
    ax_error.set_xlim(0, len(plot_df))
    ax_error.set_ylim(0, plot_df.tracking_error_dist.max() * 1.1)
    ax_error.set_title('Tracking Error Over Time')
    ax_error.set_xlabel('Frame')
    ax_error.set_ylabel('Error (pixels)')
    ax_error.grid(True)
    ax_error.legend()
    
    # Text for frame number and error info
    frame_info = ax_scatter.text(0.05, 0.95, "", transform=ax_scatter.transAxes, 
                             va='top', ha='left', bbox=dict(boxstyle="round", fc="w", alpha=0.7))
    
    # Add some basic stats to the figure
    avg_error = plot_df.tracking_error_dist.mean()
    max_error = plot_df.tracking_error_dist.max()
    in_deadzone = (plot_df.tracking_error_dist <= dead_zone_size).sum() / len(plot_df) * 100
    
    stats_text = (
        f"Total Frames: {len(df)}\n"
        f"Avg. Error: {avg_error:.1f} px\n"
        f"Max Error: {max_error:.1f} px\n"
        f"In Dead Zone: {in_deadzone:.1f}%\n"
    )
    
    ax_error.text(0.05, 0.95, stats_text, transform=ax_error.transAxes,
                 va='top', ha='left', 
                 bbox=dict(boxstyle="round", fc="w", alpha=0.7),
                 fontsize=9)
    
    # Track dot animation
    dot, = ax_error.plot([], [], 'ro')
    
    # Animation initialization function
    def init():
        scatter.set_offsets(np.array([[], []]).T)
        highlight_point.set_offsets(np.array([[], []]).T)
        line.set_data([], [])
        dot.set_data([], [])
        frame_info.set_text("")
        return scatter, highlight_point, line, dot, frame_info
    
    # Animation update function
    def update(frame_idx):
        # Get data up to current frame
        current_data = plot_df.iloc[:frame_idx+1]
        
        # Update scatter plot with all points up to current frame
        scatter.set_offsets(current_data[['tracking_error_x', 'tracking_error_y']].values)
        
        # Update current position highlight
        if frame_idx < len(plot_df):
            row = plot_df.iloc[frame_idx]
            highlight_point.set_offsets([row.tracking_error_x, row.tracking_error_y])
            
            # Update frame info
            quadrant = determine_quadrant(row.tracking_error_x, row.tracking_error_y)
            frame_info_str = (
                f"Frame: {frame_idx}\n"
                f"Error: {row.tracking_error_dist:.1f} px\n"
                f"X: {row.tracking_error_x:.1f}, Y: {row.tracking_error_y:.1f}\n"
                f"Quadrant: {quadrant}"
            )
            frame_info.set_text(frame_info_str)
        
        # Update error line plot
        x = np.arange(len(current_data))
        y = current_data.tracking_error_dist.values
        line.set_data(x, y)
        
        # Update position dot on error plot
        if frame_idx < len(plot_df):
            dot.set_data([frame_idx], [plot_df.iloc[frame_idx].tracking_error_dist])
        
        return scatter, highlight_point, line, dot, frame_info
    
    # Create animation - reduce frames if there are too many
    print(f"Creating animation with {len(plot_df)} frames...")
    
    # If there are too many frames, sample to reduce to ~300 frames
    # This makes the animation more manageable while still showing movement
    if len(plot_df) > 300:
        step = max(1, len(plot_df) // 300)
        print(f"Sampling every {step} frames to reduce animation size")
        frames = range(0, len(plot_df), step)
    else:
        frames = range(len(plot_df))
    
    anim = animation.FuncAnimation(
        fig, update, frames=frames, 
        init_func=init, blit=False, interval=50
    )
    
    # Add timestamp to figure (now this will work with the import)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    experiment_name = os.path.basename(os.path.dirname(os.path.dirname(output_path))) if '/' in output_path else 'unknown'
    fig.text(0.01, 0.01, f"Experiment: {experiment_name}\nGenerated: {timestamp}",
             fontsize=8, alpha=0.7)
    
    # Add figure title with experiment name
    fig.suptitle(f"Pan-Tilt Camera Tracking Performance\nExperiment: {experiment_name}", fontsize=14)
    
    # Save animation with correct settings for video
    print(f"Saving animation to {output_path}...")
    try:
        # Try using FFMpegWriter if available
        writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Pan-Tilt Tracker'), bitrate=1800)
        anim.save(output_path, writer=writer)
    except Exception as e:
        print(f"Error with FFMpegWriter: {e}")
        # Fall back to default writer
        try:
            anim.save(output_path, fps=20, dpi=100)
        except Exception as e:
            print(f"Error saving animation: {e}")
            # Last resort - try saving as GIF
            gif_path = output_path.replace('.mp4', '.gif')
            print(f"Trying to save as GIF: {gif_path}")
            anim.save(gif_path, writer='pillow', fps=10, dpi=80)
            print(f"Animation saved as GIF: {gif_path}")
            return
    
    print(f"Animation saved to {output_path}")
    plt.close(fig)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Pan-Tilt Tracking Visualization Tool')
    parser.add_argument('--experiment', '-e', type=str, help='Experiment name (folder in logs directory)')
    parser.add_argument('--log', '-l', type=str, help='Direct path to log file')
    parser.add_argument('--animation', '-a', action='store_true', help='Generate animation')
    parser.add_argument('--output', '-o', type=str, help='Output path for animation file')
    
    args = parser.parse_args()
    
    # Load data
    df, config = load_tracking_data(args.experiment, args.log)
    
    if df is None:
        print("Failed to load tracking data")
        return
    
    if args.animation:
        save_animation(df, args.output, config)
    else:
        interactive_visualization(df, config)

if __name__ == "__main__":
    main()
