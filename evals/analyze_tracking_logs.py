#!/usr/bin/env python3
"""
Analyze tracking logs to evaluate performance
"""

import os
import sys
import argparse
import json
import glob
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import io

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

def parse_log_file(tracking_log):
    """
    Parse log file that may be in CSV or JSON format
    
    Args:
        tracking_log: Path to log file
        
    Returns:
        pandas.DataFrame: Parsed log data
    """
    data = []
    
    with open(tracking_log, 'r') as f:
        # Read first line to determine format
        first_line = f.readline().strip()
        f.seek(0)  # Reset file pointer
        
        if first_line.startswith('{') and first_line.endswith('}'):
            # JSON format
            print("Detected JSON format log file")
            for line in f:
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError:
                    # Fallback to CSV parsing if JSON fails
                    if ',' in line:
                        print(f"Attempting to parse as CSV: {line[:50]}...")
                        try:
                            # Convert CSV line to dict
                            if not hasattr(parse_log_file, 'headers'):
                                # If this is the header row, store it
                                parse_log_file.headers = line.strip().split(',')
                                continue
                            
                            values = line.strip().split(',')
                            if len(values) == len(parse_log_file.headers):
                                entry = dict(zip(parse_log_file.headers, values))
                                
                                # Convert numeric values
                                for key, value in entry.items():
                                    try:
                                        if '.' in value:
                                            entry[key] = float(value)
                                        else:
                                            entry[key] = int(value)
                                    except (ValueError, TypeError):
                                        # Keep as string if conversion fails
                                        pass
                                
                                # Convert boolean values
                                for key in ['target_detected', 'tracking_enabled']:
                                    if key in entry:
                                        if isinstance(entry[key], str):
                                            entry[key] = entry[key].lower() == 'true'
                                
                                data.append(entry)
                        except Exception as e:
                            print(f"Error parsing line as CSV: {e}")
                    else:
                        print(f"Error parsing log entry: {line}")
        else:
            # Assume CSV format
            print("Detected CSV format log file")
            # Reset attribute to handle multiple file parses
            if hasattr(parse_log_file, 'headers'):
                delattr(parse_log_file, 'headers')
                
            # Read file using csv module for better handling of quoted fields
            csv_reader = csv.reader(f)
            
            # First row is headers
            headers = next(csv_reader)
            parse_log_file.headers = headers
            
            for row in csv_reader:
                if len(row) == len(headers):
                    entry = {}
                    for i, header in enumerate(headers):
                        value = row[i]
                        try:
                            # Convert to appropriate type
                            if value == '':
                                entry[header] = None
                            elif value.lower() == 'true':
                                entry[header] = True
                            elif value.lower() == 'false':
                                entry[header] = False
                            elif '.' in value:
                                entry[header] = float(value)
                            else:
                                try:
                                    entry[header] = int(value)
                                except ValueError:
                                    entry[header] = value
                        except Exception:
                            entry[header] = value
                    
                    data.append(entry)
                else:
                    print(f"Warning: Row has {len(row)} columns, expected {len(headers)}")
    
    if not data:
        print("No valid data found in log file")
        return None
    
    return pd.DataFrame(data)

def analyze_tracking_log(experiment_name=None, log_file=None):
    """
    Analyze tracking performance from logs
    
    Args:
        experiment_name: Name of experiment directory (e.g., "experiment_20250916_160643")
        log_file: Legacy direct log file path
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
            return
        tracking_log = os.path.join(base_dir, "logs", experiment_name, "tracking_data.log")
    
    print(f"Analyzing tracking log: {tracking_log}")
    
    # Check if log file exists
    if not os.path.exists(tracking_log):
        print(f"Error: Log file not found: {tracking_log}")
        return
    
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
        return
    
    # Basic statistics
    print("\n=== Tracking Statistics ===")
    print(f"Total frames analyzed: {len(df)}")
    
    if 'timestamp' in df.columns:
        print(f"Time period: {df.timestamp.min()} to {df.timestamp.max()}")
        print(f"Duration: {(df.timestamp.max() - df.timestamp.min()):.2f} seconds")
    
    # Detection statistics
    if 'target_detected' in df.columns:
        # Convert to boolean if it's not already
        if df.target_detected.dtype != bool:
            df['target_detected'] = df.target_detected.astype(bool)
            
        total_detections = df.target_detected.sum()
        detection_rate = total_detections / len(df) * 100
        print(f"Person detection rate: {detection_rate:.2f}% ({total_detections}/{len(df)} frames)")
    
    # Processing time statistics
    if 'processing_time_ms' in df.columns:
        avg_processing = df.processing_time_ms.mean()
        max_processing = df.processing_time_ms.max()
        fps = 1000 / avg_processing
        print(f"Average processing time: {avg_processing:.2f} ms")
        print(f"Maximum processing time: {max_processing:.2f} ms")
        print(f"Average FPS: {fps:.2f}")
    
    # Servo movement statistics
    if 'current_pan' in df.columns and 'current_tilt' in df.columns:
        # Calculate total movement
        df['pan_change'] = df.current_pan.diff().abs()
        df['tilt_change'] = df.current_tilt.diff().abs()
        
        total_pan_movement = df.pan_change.sum()
        total_tilt_movement = df.tilt_change.sum()
        
        # Calculate average speed if timestamp column exists
        if 'timestamp' in df.columns:
            avg_pan_speed = df.pan_change.mean() / ((df.timestamp.max() - df.timestamp.min()) / len(df))
            avg_tilt_speed = df.tilt_change.mean() / ((df.timestamp.max() - df.timestamp.min()) / len(df))
            print(f"Average pan speed: {avg_pan_speed:.2f} degrees/sec")
            print(f"Average tilt speed: {avg_tilt_speed:.2f} degrees/sec")
        
        print(f"Total pan movement: {total_pan_movement:.2f} degrees")
        print(f"Total tilt movement: {total_tilt_movement:.2f} degrees")
    
    # Handle different column naming conventions
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
    
    # Tracking error statistics (if available)
    if (target_x_col and target_y_col and center_x_col and center_y_col):
        # Calculate tracking error
        df['tracking_error_x'] = df[target_x_col] - df[center_x_col]
        df['tracking_error_y'] = df[target_y_col] - df[center_y_col]
        df['tracking_error_dist'] = np.sqrt(df['tracking_error_x']**2 + df['tracking_error_y']**2)
        
        # Filter out rows where target was not detected
        if 'target_detected' in df.columns:
            error_df = df[df.target_detected == True]
        else:
            error_df = df
        
        if len(error_df) > 0:
            avg_error = error_df.tracking_error_dist.mean()
            max_error = error_df.tracking_error_dist.max()
            std_error = error_df.tracking_error_dist.std()
            
            print(f"Average tracking error: {avg_error:.2f} pixels")
            print(f"Maximum tracking error: {max_error:.2f} pixels")
            print(f"Standard deviation of error: {std_error:.2f} pixels")
    
    # Create output directory for visualizations
    output_dir = os.path.join(os.path.dirname(tracking_log), "analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating visualizations in {output_dir}")
    
    # Generate visualizations
    try:
        # Plot 1: Processing time over time
        if 'processing_time_ms' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df.processing_time_ms)
            plt.axhline(y=df.processing_time_ms.mean(), color='r', linestyle='--', label=f'Mean: {df.processing_time_ms.mean():.2f} ms')
            plt.title('Processing Time per Frame')
            plt.xlabel('Frame')
            plt.ylabel('Time (ms)')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'processing_time.png'), dpi=300)
            plt.close()
        
        # Plot 2: Servo positions over time
        if 'current_pan' in df.columns and 'current_tilt' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df.current_pan, label='Pan')
            plt.plot(df.index, df.current_tilt, label='Tilt')
            plt.title('Servo Positions Over Time')
            plt.xlabel('Frame')
            plt.ylabel('Angle (degrees)')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'servo_positions.png'), dpi=300)
            plt.close()
        
        # Plot 3: Tracking error over time (if available)
        if 'tracking_error_dist' in df.columns:
            # Filter for frames with detections
            if 'target_detected' in df.columns:
                plot_df = df[df.target_detected == True]
            else:
                plot_df = df
                
            if len(plot_df) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(plot_df.index, plot_df.tracking_error_dist)
                plt.axhline(y=plot_df.tracking_error_dist.mean(), color='r', linestyle='--', label=f'Mean: {plot_df.tracking_error_dist.mean():.2f} px')
                plt.title('Tracking Error Over Time')
                plt.xlabel('Frame')
                plt.ylabel('Error (pixels)')
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(output_dir, 'tracking_error.png'), dpi=300)
                plt.close()
        
        print("Visualizations generated successfully")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"=== Tracking Analysis Summary ===\n")
        f.write(f"Log file: {tracking_log}\n")
        f.write(f"Analysis date: {datetime.datetime.now()}\n\n")
        
        f.write(f"Total frames analyzed: {len(df)}\n")
        
        if 'timestamp' in df.columns:
            f.write(f"Duration: {(df.timestamp.max() - df.timestamp.min()):.2f} seconds\n\n")
        
        if 'target_detected' in df.columns:
            f.write(f"Person detection rate: {detection_rate:.2f}% ({total_detections}/{len(df)} frames)\n")
        
        if 'processing_time_ms' in df.columns:
            f.write(f"Average processing time: {avg_processing:.2f} ms\n")
            f.write(f"Maximum processing time: {max_processing:.2f} ms\n")
            f.write(f"Average FPS: {fps:.2f}\n\n")
        
        if 'current_pan' in df.columns and 'current_tilt' in df.columns:
            f.write(f"Total pan movement: {total_pan_movement:.2f} degrees\n")
            f.write(f"Total tilt movement: {total_tilt_movement:.2f} degrees\n")
            if 'timestamp' in df.columns:
                f.write(f"Average pan speed: {avg_pan_speed:.2f} degrees/sec\n")
                f.write(f"Average tilt speed: {avg_tilt_speed:.2f} degrees/sec\n\n")
        
        if 'tracking_error_dist' in df.columns and 'target_detected' in df.columns and df[df.target_detected == True].shape[0] > 0:
            f.write(f"Average tracking error: {avg_error:.2f} pixels\n")
            f.write(f"Maximum tracking error: {max_error:.2f} pixels\n")
            f.write(f"Standard deviation of error: {std_error:.2f} pixels\n")
    
    print(f"Analysis summary saved to {summary_file}")
    print("\nEvaluation complete!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze tracking logs')
    parser.add_argument('--experiment', '-e', type=str, help='Experiment name (folder in logs directory)')
    parser.add_argument('--log', '-l', type=str, help='Direct path to log file (legacy format)')
    
    args = parser.parse_args()
    
    analyze_tracking_log(args.experiment, args.log)

if __name__ == "__main__":
    main()