#!/usr/bin/env python3
"""
Synthetic Detection Generator for Pan-Tilt Tracking Camera

This module provides synthetic human movement data for reproducible
experiments with the pan-tilt tracking camera system.

It simulates various movement patterns (linear, circular, random) and
generates Detection objects that are compatible with the YOLO tracking system.
"""

import numpy as np
import cv2
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import random


class SyntheticDetectionGenerator:
    """
    Generates synthetic detection data for benchmarking the tracking system
    """
    
    def __init__(self, frame_width=1920, frame_height=1080):
        """
        Initialize the synthetic detection generator
        
        Args:
            frame_width: Width of the synthetic frame
            frame_height: Height of the synthetic frame
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_center = (frame_width // 2, frame_height // 2)
        
        # Person size parameters (can be randomized later)
        self.person_height = int(frame_height * 0.5)  # Person takes up half the frame height
        self.person_width = int(self.person_height * 0.4)  # Person width is 40% of height
        
        # Movement parameters
        self.position = None  # Current position (center of the person)
        self.movement_pattern = None  # Current movement pattern
        self.start_time = None  # Start time for time-based movements
        self.pattern_params = {}  # Parameters for the current movement pattern
        self.current_step = 0  # Current step for step-based movements
        
    def set_movement_pattern(self, pattern: str, **kwargs):
        """
        Set the movement pattern for the synthetic person
        
        Args:
            pattern: Movement pattern ('linear', 'circular', 'random', 'zigzag', 'spiral')
            **kwargs: Pattern-specific parameters
        """
        self.movement_pattern = pattern
        self.pattern_params = kwargs
        self.start_time = time.time()
        self.current_step = 0
        
        # Initialize starting position based on pattern
        if pattern == 'linear':
            # For linear, start at one edge and move to the other
            direction = kwargs.get('direction', 'left_to_right')
            
            if direction == 'left_to_right':
                start_x = 0 + self.person_width // 2
                start_y = self.frame_height // 2
            elif direction == 'right_to_left':
                start_x = self.frame_width - self.person_width // 2
                start_y = self.frame_height // 2
            elif direction == 'top_to_bottom':
                start_x = self.frame_width // 2
                start_y = 0 + self.person_height // 2
            elif direction == 'bottom_to_top':
                start_x = self.frame_width // 2
                start_y = self.frame_height - self.person_height // 2
            else:
                # Diagonal motion
                start_x = 0 + self.person_width // 2
                start_y = 0 + self.person_height // 2
                
            self.position = (start_x, start_y)
            
        elif pattern == 'circular':
            # For circular, start at the circle perimeter
            center_x = kwargs.get('center_x', self.frame_width // 2)
            center_y = kwargs.get('center_y', self.frame_height // 2)
            radius = kwargs.get('radius', min(self.frame_width, self.frame_height) // 4)
            start_angle = kwargs.get('start_angle', 0)
            
            # Start at the perimeter
            start_x = center_x + int(radius * math.cos(math.radians(start_angle)))
            start_y = center_y + int(radius * math.sin(math.radians(start_angle)))
            self.position = (start_x, start_y)
            
        elif pattern == 'random':
            # For random, start at a random position
            self.position = (
                random.randint(self.person_width, self.frame_width - self.person_width),
                random.randint(self.person_height, self.frame_height - self.person_height)
            )
            
        elif pattern == 'zigzag':
            # For zigzag, start at one edge
            self.position = (self.person_width // 2, self.frame_height // 2)
            
        elif pattern == 'spiral':
            # For spiral, start at the center
            self.position = (self.frame_width // 2, self.frame_height // 2)
        
        else:
            # Default to center
            self.position = (self.frame_width // 2, self.frame_height // 2)
    
    def update_position(self):
        """
        Update the position based on the current movement pattern
        Returns True if the pattern is complete, False otherwise
        """
        if not self.movement_pattern or not self.position:
            return True  # No pattern set
        
        if self.movement_pattern == 'linear':
            return self._update_linear_position()
        elif self.movement_pattern == 'circular':
            return self._update_circular_position()
        elif self.movement_pattern == 'random':
            return self._update_random_position()
        elif self.movement_pattern == 'zigzag':
            return self._update_zigzag_position()
        elif self.movement_pattern == 'spiral':
            return self._update_spiral_position()
        
        return True  # Unknown pattern
    
    def _update_linear_position(self):
        """Update position for linear movement"""
        direction = self.pattern_params.get('direction', 'left_to_right')
        speed = self.pattern_params.get('speed', 5)  # pixels per update
        
        x, y = self.position
        
        # Calculate new position
        if direction == 'left_to_right':
            x += speed
            # Check if we've reached the end
            if x >= self.frame_width - self.person_width // 2:
                return True
        elif direction == 'right_to_left':
            x -= speed
            if x <= self.person_width // 2:
                return True
        elif direction == 'top_to_bottom':
            y += speed
            if y >= self.frame_height - self.person_height // 2:
                return True
        elif direction == 'bottom_to_top':
            y -= speed
            if y <= self.person_height // 2:
                return True
        elif direction == 'diagonal_down':
            x += speed
            y += speed
            if x >= self.frame_width - self.person_width // 2 or y >= self.frame_height - self.person_height // 2:
                return True
        elif direction == 'diagonal_up':
            x += speed
            y -= speed
            if x >= self.frame_width - self.person_width // 2 or y <= self.person_height // 2:
                return True
        
        self.position = (x, y)
        return False
    
    def _update_circular_position(self):
        """Update position for circular movement"""
        center_x = self.pattern_params.get('center_x', self.frame_width // 2)
        center_y = self.pattern_params.get('center_y', self.frame_height // 2)
        radius = self.pattern_params.get('radius', min(self.frame_width, self.frame_height) // 4)
        speed = self.pattern_params.get('speed', 2)  # degrees per update
        clockwise = self.pattern_params.get('clockwise', True)
        complete_circle = self.pattern_params.get('complete_circle', True)
        
        # Calculate angle based on time or steps
        if 'duration' in self.pattern_params:
            # Time-based: complete the circle in the given duration
            duration = self.pattern_params['duration']
            elapsed = time.time() - self.start_time
            angle = (elapsed / duration) * 360.0
            if elapsed >= duration and complete_circle:
                return True
        else:
            # Step-based: update by a fixed angle each time
            self.current_step += 1
            angle = self.current_step * speed
            if angle >= 360.0 and complete_circle:
                return True
        
        # Apply direction
        if not clockwise:
            angle = -angle
        
        # Calculate new position
        start_angle = self.pattern_params.get('start_angle', 0)
        angle = start_angle + angle
        
        x = center_x + int(radius * math.cos(math.radians(angle)))
        y = center_y + int(radius * math.sin(math.radians(angle)))
        
        self.position = (x, y)
        return False
    
    def _update_random_position(self):
        """Update position for random movement"""
        max_steps = self.pattern_params.get('max_steps', 100)
        max_speed = self.pattern_params.get('max_speed', 10)
        
        x, y = self.position
        
        # Random direction and speed
        dx = random.randint(-max_speed, max_speed)
        dy = random.randint(-max_speed, max_speed)
        
        # Ensure we stay within bounds
        x = max(self.person_width // 2, min(x + dx, self.frame_width - self.person_width // 2))
        y = max(self.person_height // 2, min(y + dy, self.frame_height - self.person_height // 2))
        
        self.position = (x, y)
        
        # Check if we've reached the maximum number of steps
        self.current_step += 1
        return self.current_step >= max_steps
    
    def _update_zigzag_position(self):
        """Update position for zigzag movement"""
        speed = self.pattern_params.get('speed', 5)  # pixels per update
        amplitude = self.pattern_params.get('amplitude', self.frame_height // 4)
        frequency = self.pattern_params.get('frequency', 0.01)  # lower = wider zigzag
        
        self.current_step += 1
        
        # Calculate new x position (constant speed across screen)
        x = self.current_step * speed + self.person_width // 2
        
        # Calculate new y position (zigzag pattern)
        y = self.frame_height // 2 + amplitude * math.sin(frequency * x)
        
        # Check if we've reached the end of the frame
        if x >= self.frame_width - self.person_width // 2:
            return True
            
        self.position = (int(x), int(y))
        return False
    
    def _update_spiral_position(self):
        """Update position for spiral movement"""
        speed = self.pattern_params.get('speed', 1)  # degrees per update
        growth_rate = self.pattern_params.get('growth_rate', 0.2)  # spiral growth factor
        max_radius = self.pattern_params.get('max_radius', min(self.frame_width, self.frame_height) // 3)
        center_x = self.pattern_params.get('center_x', self.frame_width // 2)
        center_y = self.pattern_params.get('center_y', self.frame_height // 2)
        
        self.current_step += speed
        
        # Calculate radius based on angle
        angle = self.current_step
        radius = growth_rate * angle
        
        # Check if we've reached the maximum radius
        if radius >= max_radius:
            return True
            
        # Calculate new position
        x = center_x + int(radius * math.cos(math.radians(angle)))
        y = center_y + int(radius * math.sin(math.radians(angle)))
        
        self.position = (x, y)
        return False
    
    def get_detection(self):
        """
        Generate a detection object based on the current position
        Returns None if no valid position
        """
        if not self.position:
            return None
            
        x, y = self.position
        
        # Calculate bounding box
        x1 = max(0, x - self.person_width // 2)
        y1 = max(0, y - self.person_height // 2)
        x2 = min(self.frame_width, x + self.person_width // 2)
        y2 = min(self.frame_height, y + self.person_height // 2)
        
        # Create a detection object that matches the format expected by the tracker
        detection = {
            'class_id': 0,
            'class_name': 'person',
            'confidence': random.uniform(0.75, 0.95),  # Simulate some confidence variation
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'center': (int(x), int(y)),
            'area': int((x2 - x1) * (y2 - y1))
        }
        
        return detection
    
    def create_synthetic_frame(self, background_color=(0, 0, 0)):
        """
        Create a synthetic frame with the person rendered at the current position
        
        Args:
            background_color: RGB color for the background
        
        Returns:
            Numpy array representing the synthetic frame
        """
        # Create a blank frame
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        frame[:, :] = background_color
        
        # If we have a valid position, draw a person
        if self.position:
            x, y = self.position
            x1 = max(0, x - self.person_width // 2)
            y1 = max(0, y - self.person_height // 2)
            x2 = min(self.frame_width, x + self.person_width // 2)
            y2 = min(self.frame_height, y + self.person_height // 2)
            
            # Draw a simple person shape
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Body
            
            # Draw head
            head_size = self.person_width // 2
            head_y = y1 + head_size // 2
            cv2.circle(frame, (x, head_y), head_size, (0, 0, 200), -1)
            
            # Draw center point
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        
        return frame