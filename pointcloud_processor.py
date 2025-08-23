#!/usr/bin/env python3
"""
Point Cloud Processing Module
Handles 3D point cloud data from ZED camera for object localization and mapping
"""

import numpy as np
import cv2
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import json

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not available - using numpy-based processing")

@dataclass
class PointCloudRegion:
    """3D region of interest in point cloud"""
    center: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]  # width, height, depth
    point_count: int
    density: float
    bbox_2d: Tuple[int, int, int, int]  # corresponding 2D bounding box
    confidence: float

class PointCloudProcessor:
    def __init__(self, 
                 max_depth: float = 10.0,
                 min_depth: float = 0.3,
                 voxel_size: float = 0.05,
                 noise_threshold: int = 10):
        
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.voxel_size = voxel_size
        self.noise_threshold = noise_threshold
        
        # Point cloud history for temporal filtering
        self.point_cloud_history: List[np.ndarray] = []
        self.max_history_size = 5
        
        print(f"Point cloud processor initialized (Open3D: {OPEN3D_AVAILABLE})")

    def filter_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Filter and clean point cloud data"""
        if len(points) == 0:
            return points
        
        # Remove invalid points (NaN, inf)
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        
        if len(points) == 0:
            return points
        
        # Filter by depth range
        depths = points[:, 2]  # Z coordinate is depth
        depth_mask = (depths >= self.min_depth) & (depths <= self.max_depth)
        points = points[depth_mask]
        
        return points

    def extract_object_pointcloud(self, 
                                 point_cloud: np.ndarray, 
                                 bbox_2d: Tuple[int, int, int, int],
                                 image_width: int, 
                                 image_height: int,
                                 expansion_factor: float = 1.2) -> Tuple[np.ndarray, PointCloudRegion]:
        """Extract point cloud within 2D bounding box region"""
        x1, y1, x2, y2 = bbox_2d
        
        # Expand bounding box slightly
        width = x2 - x1
        height = y2 - y1
        expand_w = int(width * (expansion_factor - 1) / 2)
        expand_h = int(height * (expansion_factor - 1) / 2)
        
        x1_exp = max(0, x1 - expand_w)
        y1_exp = max(0, y1 - expand_h)
        x2_exp = min(image_width, x2 + expand_w)
        y2_exp = min(image_height, y2 + expand_h)
        
        # Extract points within bounding box
        object_points = []
        for y in range(y1_exp, y2_exp):
            for x in range(x1_exp, x2_exp):
                if 0 <= y < point_cloud.shape[0] and 0 <= x < point_cloud.shape[1]:
                    point = point_cloud[y, x]
                    if np.isfinite(point).all() and point[2] > 0:  # Valid point
                        object_points.append(point)
        
        if not object_points:
            return np.array([]), None
        
        object_points = np.array(object_points)
        object_points = self.filter_point_cloud(object_points)
        
        if len(object_points) == 0:
            return object_points, None
        
        # Calculate region statistics
        center = np.mean(object_points, axis=0)
        min_bounds = np.min(object_points, axis=0)
        max_bounds = np.max(object_points, axis=0)
        dimensions = max_bounds - min_bounds
        
        # Calculate density (points per cubic meter)
        volume = np.prod(dimensions)
        density = len(object_points) / max(volume, 0.001)
        
        region = PointCloudRegion(
            center=tuple(center),
            dimensions=tuple(dimensions),
            point_count=len(object_points),
            density=density,
            bbox_2d=(x1_exp, y1_exp, x2_exp, y2_exp),
            confidence=min(1.0, len(object_points) / 1000.0)  # Confidence based on point count
        )
        
        return object_points, region

    def cluster_points(self, points: np.ndarray, eps: float = 0.1, min_samples: int = 10) -> List[np.ndarray]:
        """Cluster point cloud using DBSCAN-like algorithm"""
        if len(points) == 0:
            return []
        
        if OPEN3D_AVAILABLE:
            # Use Open3D clustering
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples))
            
            clusters = []
            for cluster_id in range(labels.max() + 1):
                cluster_mask = labels == cluster_id
                if np.sum(cluster_mask) >= min_samples:
                    clusters.append(points[cluster_mask])
            
            return clusters
        else:
            # Simple numpy-based clustering
            return self._simple_cluster(points, eps, min_samples)

    def _simple_cluster(self, points: np.ndarray, eps: float, min_samples: int) -> List[np.ndarray]:
        """Simple clustering algorithm using numpy"""
        clusters = []
        visited = np.zeros(len(points), dtype=bool)
        
        for i, point in enumerate(points):
            if visited[i]:
                continue
            
            # Find neighbors
            distances = np.linalg.norm(points - point, axis=1)
            neighbors = np.where(distances < eps)[0]
            
            if len(neighbors) >= min_samples:
                # Start new cluster
                cluster_points = []
                queue = list(neighbors)
                
                while queue:
                    idx = queue.pop(0)
                    if visited[idx]:
                        continue
                    
                    visited[idx] = True
                    cluster_points.append(points[idx])
                    
                    # Find neighbors of this point
                    distances = np.linalg.norm(points - points[idx], axis=1)
                    new_neighbors = np.where(distances < eps)[0]
                    
                    if len(new_neighbors) >= min_samples:
                        queue.extend([n for n in new_neighbors if not visited[n]])
                
                if len(cluster_points) >= min_samples:
                    clusters.append(np.array(cluster_points))
        
        return clusters

    def estimate_object_pose(self, points: np.ndarray) -> Dict:
        """Estimate 6DOF pose of object from point cloud"""
        if len(points) < 3:
            return {}
        
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        
        # Calculate principal axes using PCA
        centered_points = points - centroid
        covariance = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # Sort by eigenvalue (largest first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate bounding box dimensions along principal axes
        projected = centered_points @ eigenvectors
        dimensions = np.max(projected, axis=0) - np.min(projected, axis=0)
        
        return {
            'position': centroid,
            'orientation': eigenvectors,
            'dimensions': dimensions,
            'eigenvalues': eigenvalues,
            'volume_estimate': np.prod(dimensions)
        }

    def track_objects_temporal(self, current_regions: List[PointCloudRegion]) -> List[PointCloudRegion]:
        """Track objects across multiple frames using temporal consistency"""
        # Simple temporal filtering - average positions over recent frames
        # In a full implementation, you'd use Kalman filters or particle filters
        
        if not hasattr(self, 'previous_regions'):
            self.previous_regions = []
        
        tracked_regions = []
        
        for current_region in current_regions:
            # Find closest previous region
            best_match = None
            min_distance = float('inf')
            
            for prev_region in self.previous_regions:
                distance = np.linalg.norm(
                    np.array(current_region.center) - np.array(prev_region.center)
                )
                if distance < min_distance and distance < 0.5:  # 50cm matching threshold
                    min_distance = distance
                    best_match = prev_region
            
            if best_match:
                # Smooth the position
                alpha = 0.7  # Smoothing factor
                smoothed_center = (
                    alpha * np.array(current_region.center) + 
                    (1 - alpha) * np.array(best_match.center)
                )
                
                # Create updated region
                updated_region = PointCloudRegion(
                    center=tuple(smoothed_center),
                    dimensions=current_region.dimensions,
                    point_count=current_region.point_count,
                    density=current_region.density,
                    bbox_2d=current_region.bbox_2d,
                    confidence=min(1.0, current_region.confidence + 0.1)
                )
                tracked_regions.append(updated_region)
            else:
                tracked_regions.append(current_region)
        
        self.previous_regions = tracked_regions[-10:]  # Keep last 10 frames
        return tracked_regions

    def save_point_cloud(self, points: np.ndarray, filename: str):
        """Save point cloud to file"""
        if OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(filename, pcd)
        else:
            # Save as numpy array
            np.save(filename.replace('.ply', '.npy'), points)
        
        print(f"Saved point cloud with {len(points)} points to {filename}")

    def create_occupancy_map(self, 
                           points: np.ndarray, 
                           resolution: float = 0.1,
                           map_size: Tuple[float, float] = (20.0, 20.0)) -> np.ndarray:
        """Create 2D occupancy map from point cloud (bird's eye view)"""
        if len(points) == 0:
            return np.zeros((int(map_size[1]/resolution), int(map_size[0]/resolution)))
        
        # Filter points to reasonable height range (remove floor/ceiling)
        height_filtered = points[(points[:, 1] > -1.5) & (points[:, 1] < 2.0)]
        
        if len(height_filtered) == 0:
            return np.zeros((int(map_size[1]/resolution), int(map_size[0]/resolution)))
        
        # Project to 2D (X-Z plane, removing Y)
        points_2d = height_filtered[:, [0, 2]]  # X and Z coordinates
        
        # Create occupancy grid
        map_width = int(map_size[0] / resolution)
        map_height = int(map_size[1] / resolution)
        occupancy_map = np.zeros((map_height, map_width))
        
        # Convert points to grid coordinates
        grid_x = ((points_2d[:, 0] + map_size[0]/2) / resolution).astype(int)
        grid_z = ((points_2d[:, 1]) / resolution).astype(int)
        
        # Filter points within map bounds
        valid_mask = (
            (grid_x >= 0) & (grid_x < map_width) &
            (grid_z >= 0) & (grid_z < map_height)
        )
        
        valid_x = grid_x[valid_mask]
        valid_z = grid_z[valid_mask]
        
        # Mark occupied cells
        for x, z in zip(valid_x, valid_z):
            occupancy_map[z, x] = 1
        
        return occupancy_map

    def visualize_point_cloud_2d(self, points: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Create 2D visualization of point cloud overlaid on image"""
        vis = np.zeros((*image_shape, 3), dtype=np.uint8)
        
        if len(points) == 0:
            return vis
        
        # Color points by depth
        depths = points[:, 2]
        if len(depths) > 0:
            min_depth, max_depth = np.percentile(depths, [5, 95])  # Remove outliers
            normalized_depths = np.clip((depths - min_depth) / (max_depth - min_depth), 0, 1)
            
            # Apply colormap (blue=close, red=far)
            colors = plt.cm.turbo(normalized_depths)[:, :3] * 255
            
            # Project 3D points to 2D image coordinates (simplified)
            # This would need proper camera calibration in a real implementation
            fx, fy = 700, 700  # Approximate focal lengths for ZED2 HD
            cx, cy = image_shape[1]//2, image_shape[0]//2
            
            image_x = (points[:, 0] * fx / points[:, 2] + cx).astype(int)
            image_y = (points[:, 1] * fy / points[:, 2] + cy).astype(int)
            
            # Filter points within image bounds
            valid_mask = (
                (image_x >= 0) & (image_x < image_shape[1]) &
                (image_y >= 0) & (image_y < image_shape[0])
            )
            
            valid_x = image_x[valid_mask]
            valid_y = image_y[valid_mask]
            valid_colors = colors[valid_mask]
            
            # Draw points
            for x, y, color in zip(valid_x, valid_y, valid_colors):
                cv2.circle(vis, (x, y), 2, color.astype(int).tolist(), -1)
        
        return vis


def test_pointcloud_processing():
    """Test point cloud processing functions"""
    processor = PointCloudProcessor()
    
    # Generate synthetic point cloud data
    print("Generating synthetic point cloud...")
    n_points = 10000
    points = np.random.randn(n_points, 3)
    points[:, 2] = np.abs(points[:, 2]) + 1  # Ensure positive depth
    
    print(f"Generated {len(points)} points")
    
    # Test filtering
    filtered_points = processor.filter_point_cloud(points)
    print(f"After filtering: {len(filtered_points)} points")
    
    # Test clustering
    clusters = processor.cluster_points(filtered_points[:1000])  # Use subset for speed
    print(f"Found {len(clusters)} clusters")
    
    # Test pose estimation
    if clusters:
        pose = processor.estimate_object_pose(clusters[0])
        print(f"Largest cluster pose: center={pose.get('position', 'N/A')}")
    
    # Test occupancy mapping
    occupancy_map = processor.create_occupancy_map(filtered_points)
    print(f"Occupancy map shape: {occupancy_map.shape}")
    print(f"Occupied cells: {np.sum(occupancy_map)}")


if __name__ == "__main__":
    test_pointcloud_processing()