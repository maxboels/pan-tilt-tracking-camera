#!/usr/bin/env python3
"""
Tracking Log Analysis Tool
Analyzes tracking log files to evaluate system performance
"""

import os
import sys
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.gridspec import GridSpec
import glob


def find_latest_log():
    """Find the most recent log file in the logs directory"""
    # Look for experiment folders
    experiment_dirs = [d for d in glob.glob(os.path.join('logs', '*')) 
                      if os.path.isdir(d)]
    
    if not experiment_dirs:
        return None
    
    # Sort by modification time (newest first)
    experiment_dirs.sort(key=os.path.getmtime, reverse=True)
    
    # Look for tracking_data.log in the most recent experiment folder
    latest_dir = experiment_dirs[0]
    log_file = os.path.join(latest_dir, 'tracking_data.log')
    
    if os.path.exists(log_file):
        return log_file
    
    # Fall back to old-style logs if no experiment folders contain logs
    legacy_logs = glob.glob(os.path.join('logs', 'tracking_run_*.log'))
    if legacy_logs:
        legacy_logs.sort(key=os.path.getmtime, reverse=True)
        return legacy_logs[0]
    
    return None


def analyze_log(log_file):
    """Analyze tracking log file and generate visualizations"""
    print(f"Analyzing log file: {log_file}")
    
    # Load the log data
    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error loading log file: {e}")
        return
    
    # Basic statistics
    total_frames = len(df)
    detection_frames = df['target_detected'].sum()
    avg_processing_time = df['processing_time_ms'].mean()
    avg_distance_error = df[df['target_detected']]['distance_error'].mean()
    
    print(f"Total frames: {total_frames}")
    print(f"Frames with detections: {detection_frames} ({detection_frames/total_frames*100:.1f}%)")
    print(f"Average processing time: {avg_processing_time:.2f} ms")
    print(f"Average distance error: {avg_distance_error:.2f} pixels")
    
    # Create a figure with subplots
    plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2)
    
    # Plot 1: Distance Error Over Time
    ax1 = plt.subplot(gs[0, 0])
    df_detected = df[df['target_detected']]
    ax1.plot(df_detected['frame_number'], df_detected['distance_error'])
    ax1.set_title('Distance Error Over Time')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Error (pixels)')
    ax1.grid(True)
    
    # Plot 2: Servo Positions
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(df['frame_number'], df['current_pan'], label='Pan')
    ax2.plot(df['frame_number'], df['current_tilt'], label='Tilt')
    ax2.set_title('Servo Positions Over Time')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Angle (degrees)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Target Position Heatmap
    ax3 = plt.subplot(gs[1, :])
    detected = df[df['target_detected']]
    if not detected.empty:
        heatmap, xedges, yedges = np.histogram2d(
            detected['target_center_x'], 
            detected['target_center_y'],
            bins=50
        )
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax3.imshow(heatmap.T, origin='lower', extent=extent, cmap='hot')
        ax3.set_title('Target Position Heatmap')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
    else:
        ax3.text(0.5, 0.5, "No detections in log data", 
                horizontalalignment='center', verticalalignment='center')
    
    # Plot 4: Processing Time
    ax4 = plt.subplot(gs[2, 0])
    ax4.plot(df['frame_number'], df['processing_time_ms'])
    ax4.set_title('Processing Time')
    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('Time (ms)')
    ax4.grid(True)
    
    # Plot 5: Error Distribution
    ax5 = plt.subplot(gs[2, 1])
    if not df_detected.empty:
        ax5.hist(df_detected['distance_error'], bins=30)
        ax5.set_title('Error Distribution')
        ax5.set_xlabel('Distance Error (pixels)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True)
    else:
        ax5.text(0.5, 0.5, "No detections in log data", 
                horizontalalignment='center', verticalalignment='center')
    
    # Add overall title
    plt.suptitle(f"Pan-Tilt Camera Tracking Analysis - {os.path.basename(log_file)}", fontsize=16)
    plt.tight_layout()
    
    # Save the figure to the same directory as the log file
    log_dir = os.path.dirname(log_file)
    output_file = os.path.join(log_dir, "analysis.png")
    plt.savefig(output_file, dpi=150)
    print(f"Analysis saved to: {output_file}")
    
    # Show the figure
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze tracking log files')
    parser.add_argument('--log', '-l', type=str, help='Path to log file')
    parser.add_argument('--experiment', '-e', type=str, help='Experiment directory name')
    args = parser.parse_args()
    
    # If experiment is specified, look for tracking_data.log in that experiment folder
    log_file = None
    if args.experiment:
        experiment_path = os.path.join('logs', args.experiment)
        if os.path.isdir(experiment_path):
            log_file = os.path.join(experiment_path, 'tracking_data.log')
            if not os.path.exists(log_file):
                print(f"No tracking_data.log found in experiment '{args.experiment}'")
                return
        else:
            print(f"Experiment folder '{args.experiment}' not found")
            return
    # If log file is directly specified
    elif args.log:
        log_file = args.log
    # Otherwise, use the latest one
    else:
        log_file = find_latest_log()
        if not log_file:
            print("No log files found. Please specify a log file with --log or an experiment with --experiment.")
            return
    
    analyze_log(log_file)


if __name__ == "__main__":
    main()