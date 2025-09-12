#!/usr/bin/env python3
"""
Simple Performance Logger for ZED2 Tracking System
Lightweight logging to identify bottlenecks
"""

import time
import csv
import psutil
from contextlib import contextmanager
from datetime import datetime

class Timer:
    """Simple timer context manager for measuring execution time"""
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
    
    def ms(self):
        """Return elapsed time in milliseconds"""
        return (self.end_time - self.start_time) * 1000

class SimplePerformanceLogger:
    """Lightweight performance logger for tracking system bottlenecks"""
    
    def __init__(self, csv_filename=None):
        if csv_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"tracking_performance_{timestamp}.csv"
        
        self.csv_file = csv_filename
        self.fieldnames = [
            'frame', 'timestamp', 'total_ms', 'zed_grab_ms', 'yolo_ms', 
            'depth_ms', 'servo_ms', 'viz_ms', 'detections_count', 
            'target_found', 'cpu_percent', 'memory_mb'
        ]
        
        # Initialize CSV file
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        
        print(f"Performance logging to: {self.csv_file}")
    
    def log_frame(self, frame_num, timings, detections, target_found):
        """Log performance data for a single frame"""
        row = {
            'frame': frame_num,
            'timestamp': time.time(),
            'total_ms': timings.get('total', 0),
            'zed_grab_ms': timings.get('zed_grab', 0),
            'yolo_ms': timings.get('yolo', 0),
            'depth_ms': timings.get('depth', 0),
            'servo_ms': timings.get('servo', 0),
            'viz_ms': timings.get('viz', 0),
            'detections_count': len(detections),
            'target_found': target_found,
            'cpu_percent': psutil.cpu_percent(),
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)