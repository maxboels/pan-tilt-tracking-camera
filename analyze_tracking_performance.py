#!/usr/bin/env python3
"""
Performance Analysis Script for Pan-Tilt Tracking Camera
Analyzes tracking errors and provides recommendations for improvement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import os

class TrackingPerformanceAnalyzer:
    def __init__(self, csv_file=None):
        self.csv_file = csv_file
        self.data = None
        
    def load_data(self):
        """Load tracking data from CSV file"""
        if self.csv_file and os.path.exists(self.csv_file):
            self.data = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.data)} tracking records")
        else:
            print("No CSV file found, analyzing current session errors")
            
    def analyze_errors(self, pan_errors, tilt_errors):
        """Analyze pan and tilt tracking errors"""
        pan_errors = np.array(pan_errors)
        tilt_errors = np.array(tilt_errors)
        
        analysis = {
            'pan_stats': {
                'mean': np.mean(pan_errors),
                'std': np.std(pan_errors),
                'max': np.max(pan_errors),
                'min': np.min(pan_errors)
            },
            'tilt_stats': {
                'mean': np.mean(tilt_errors),
                'std': np.std(tilt_errors),
                'max': np.max(tilt_errors),
                'min': np.min(tilt_errors)
            }
        }
        
        return analysis
        
    def generate_recommendations(self, analysis):
        """Generate recommendations based on error analysis"""
        recommendations = []
        
        pan_mean = abs(analysis['pan_stats']['mean'])
        tilt_mean = abs(analysis['tilt_stats']['mean'])
        pan_std = analysis['pan_stats']['std']
        tilt_std = analysis['tilt_stats']['std']
        
        # High error recommendations
        if pan_mean > 500:
            recommendations.append("HIGH PAN ERROR: Consider reducing pan motor speed or increasing P gain")
        if tilt_mean > 200:
            recommendations.append("HIGH TILT ERROR: Consider reducing tilt motor speed or increasing P gain")
            
        # Stability recommendations
        if pan_std > 200:
            recommendations.append("PAN INSTABILITY: Add damping (D gain) or reduce I gain to prevent oscillation")
        if tilt_std > 100:
            recommendations.append("TILT INSTABILITY: Add damping (D gain) or reduce I gain to prevent oscillation")
            
        # General recommendations
        if pan_mean > 100 or tilt_mean > 50:
            recommendations.append("Consider implementing adaptive PID gains based on error magnitude")
            recommendations.append("Check camera calibration and coordinate system alignment")
            
        return recommendations
        
    def plot_errors(self, pan_errors, tilt_errors, save_path=None):
        """Plot tracking errors over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Pan errors
        ax1.plot(pan_errors, 'b-', label='Pan Error')
        ax1.set_ylabel('Pan Error (pixels)')
        ax1.set_title('Pan Tracking Errors Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Tilt errors
        ax2.plot(tilt_errors, 'r-', label='Tilt Error')
        ax2.set_ylabel('Tilt Error (pixels)')
        ax2.set_xlabel('Time Steps')
        ax2.set_title('Tilt Tracking Errors Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze pan-tilt tracking performance')
    parser.add_argument('--csv', type=str, help='Path to CSV file with tracking data')
    args = parser.parse_args()
    
    # Sample data from your session
    pan_errors = [801, 801, 893, 893, 893, 893, 893, 893, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 686, 686, 686, 686, 686, 686, 707, 684, 687]
    tilt_errors = [299, 299, 213, 213, 213, 213, 213, 213, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, 251, 251, 251, 251, 251, 251, 266, 257, 242]
    
    analyzer = TrackingPerformanceAnalyzer(args.csv)
    
    # Analyze errors
    analysis = analyzer.analyze_errors(pan_errors, tilt_errors)
    
    # Print analysis results
    print("\n=== TRACKING PERFORMANCE ANALYSIS ===")
    print(f"Pan Error - Mean: {analysis['pan_stats']['mean']:.1f}, Std: {analysis['pan_stats']['std']:.1f}")
    print(f"Tilt Error - Mean: {analysis['tilt_stats']['mean']:.1f}, Std: {analysis['tilt_stats']['std']:.1f}")
    
    # Generate and display recommendations
    recommendations = analyzer.generate_recommendations(analysis)
    print("\n=== RECOMMENDATIONS ===")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Plot errors
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"/home/maxboels/projects/pan-tilt-tracking-camera/tracking_errors_{timestamp}.png"
    analyzer.plot_errors(pan_errors, tilt_errors, plot_path)

if __name__ == "__main__":
    main()