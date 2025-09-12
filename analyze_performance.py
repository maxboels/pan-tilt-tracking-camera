#!/usr/bin/env python3
"""
Performance Analysis Script for Pan-Tilt Tracking Camera
Analyzes tracking performance data and generates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class PerformanceAnalyzer:
    def __init__(self, csv_file):
        """Initialize the performance analyzer with CSV data"""
        self.csv_file = csv_file
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the CSV data"""
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.data)} records from {self.csv_file}")
            print(f"Columns: {list(self.data.columns)}")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def analyze_errors(self):
        """Analyze pan and tilt errors"""
        if self.data is None:
            print("No data loaded")
            return
        
        # Assuming columns exist for pan and tilt errors
        error_cols = [col for col in self.data.columns if 'error' in col.lower()]
        
        if not error_cols:
            print("No error columns found in data")
            return
        
        print("\n=== Error Analysis ===")
        for col in error_cols:
            print(f"\n{col}:")
            print(f"  Mean: {self.data[col].mean():.2f}")
            print(f"  Std:  {self.data[col].std():.2f}")
            print(f"  Min:  {self.data[col].min():.2f}")
            print(f"  Max:  {self.data[col].max():.2f}")
    
    def plot_error_trends(self):
        """Plot error trends over time"""
        if self.data is None:
            return
        
        error_cols = [col for col in self.data.columns if 'error' in col.lower()]
        
        if not error_cols:
            print("No error columns found for plotting")
            return
        
        plt.figure(figsize=(12, 8))
        
        for i, col in enumerate(error_cols[:2]):  # Limit to first 2 error columns
            plt.subplot(2, 1, i + 1)
            plt.plot(self.data.index, self.data[col], alpha=0.7, label=col)
            plt.title(f'{col} Over Time')
            plt.xlabel('Sample Index')
            plt.ylabel('Error Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('/home/maxboels/projects/pan-tilt-tracking-camera/error_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Error trends plot saved as 'error_trends.png'")
    
    def plot_error_distribution(self):
        """Plot error distribution histograms"""
        if self.data is None:
            return
        
        error_cols = [col for col in self.data.columns if 'error' in col.lower()]
        
        if not error_cols:
            return
        
        plt.figure(figsize=(12, 6))
        
        for i, col in enumerate(error_cols[:2]):
            plt.subplot(1, 2, i + 1)
            plt.hist(self.data[col], bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'{col} Distribution')
            plt.xlabel('Error Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/maxboels/projects/pan-tilt-tracking-camera/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Error distribution plot saved as 'error_distribution.png'")
    
    def calculate_stability_metrics(self):
        """Calculate tracking stability metrics"""
        if self.data is None:
            return
        
        error_cols = [col for col in self.data.columns if 'error' in col.lower()]
        
        print("\n=== Stability Metrics ===")
        for col in error_cols:
            # Calculate consecutive error differences
            error_diff = self.data[col].diff().abs()
            stability = 1 / (1 + error_diff.mean())  # Higher value = more stable
            
            print(f"\n{col}:")
            print(f"  Average error change: {error_diff.mean():.2f}")
            print(f"  Stability score: {stability:.3f}")
            print(f"  Values within ±50: {(abs(self.data[col]) <= 50).sum()}/{len(self.data)} ({(abs(self.data[col]) <= 50).mean()*100:.1f}%)")
    
    def generate_report(self):
        """Generate a comprehensive performance report"""
        print("="*50)
        print("PAN-TILT TRACKING PERFORMANCE REPORT")
        print("="*50)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data file: {self.csv_file}")
        
        if self.data is not None:
            print(f"Total samples: {len(self.data)}")
            
            # Run all analyses
            self.analyze_errors()
            self.calculate_stability_metrics()
            
            # Generate plots
            self.plot_error_trends()
            self.plot_error_distribution()
            
            print("\n=== Recommendations ===")
            error_cols = [col for col in self.data.columns if 'error' in col.lower()]
            for col in error_cols:
                avg_error = abs(self.data[col]).mean()
                if avg_error > 100:
                    print(f"• {col}: High average error ({avg_error:.1f}) - Consider tuning PID parameters")
                elif avg_error > 50:
                    print(f"• {col}: Moderate error ({avg_error:.1f}) - Minor adjustments needed")
                else:
                    print(f"• {col}: Good performance ({avg_error:.1f}) - System is tracking well")

def main():
    # Look for CSV files in the current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in current directory")
        return
    
    print("Available CSV files:")
    for i, f in enumerate(csv_files):
        print(f"{i+1}. {f}")
    
    if len(csv_files) == 1:
        selected_file = csv_files[0]
    else:
        try:
            choice = int(input("Select file (number): ")) - 1
            selected_file = csv_files[choice]
        except (ValueError, IndexError):
            print("Invalid selection")
            return
    
    print(f"Analyzing: {selected_file}")
    analyzer = PerformanceAnalyzer(selected_file)
    analyzer.generate_report()

if __name__ == "__main__":
    main()