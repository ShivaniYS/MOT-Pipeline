"""
Utility functions for object detection and tracking pipeline.
Provides helper functions for visualization, CSV export, ROI handling, etc.
"""

import cv2
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import random

from config import Config
from tracker import KalmanBoxTracker

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT
    )


def generate_colors():
    """Generate distinct colors for different track IDs."""
    random.seed(42)  # For reproducible colors
    colors = {}
    
    def get_color(track_id: int) -> Tuple[int, int, int]:
        if track_id not in colors:
            # Generate random color with good visibility
            hue = random.randint(0, 180)
            saturation = 200
            value = 200
            
            # Convert HSV to BGR
            color = cv2.cvtColor(
                np.uint8([[[hue, saturation, value]]]), 
                cv2.COLOR_HSV2BGR
            )[0][0]
            
            colors[track_id] = tuple(map(int, color))
        
        return colors[track_id]
    
    return get_color


def draw_detections(frame: np.ndarray, 
                   tracks: Dict[int, Dict],
                   velocities: Dict[int, Optional[float]]) -> np.ndarray:
    """
    Draw bounding boxes, labels, and velocity estimates on frame.
    
    Args:
        frame: Input BGR frame
        tracks: Dictionary of active tracks from tracker
        velocities: Dictionary mapping track_id to estimated velocity
        
    Returns:
        Annotated frame
    """
    display_frame = frame.copy()
    get_color = generate_colors()
    
    for track_id, track_info in tracks.items():
        bbox = track_info['bbox']
        class_name = track_info['class_name']
        confidence = track_info['confidence']
        
        # Get color for this track
        color = get_color(track_id)
        
        # Draw bounding box
        cv2.rectangle(
            display_frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color,
            Config.BOX_THICKNESS
        )
        
        # Prepare label
        velocity = velocities.get(track_id)
        if velocity is not None:
            velocity_text = VelocityEstimator.format_velocity(velocity)
            label = f"{class_name} | ID:{track_id} | {velocity_text}"
        else:
            label = f"{class_name} | ID:{track_id}"
        
        # Draw label background
        label_size = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            Config.FONT_SCALE, 
            Config.TEXT_THICKNESS
        )[0]
        
        cv2.rectangle(
            display_frame,
            (bbox[0], bbox[1] - label_size[1] - 5),
            (bbox[0] + label_size[0], bbox[1]),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            display_frame,
            label,
            (bbox[0], bbox[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            Config.FONT_SCALE,
            (255, 255, 255),
            Config.TEXT_THICKNESS,
            cv2.LINE_AA
        )
        
        # Draw center point
        center = track_info['center']
        cv2.circle(display_frame, center, 3, (0, 255, 0), -1)
    
    # Add frame info
    cv2.putText(
        display_frame,
        f"Active tracks: {len(tracks)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    
    return display_frame


def export_to_csv(tracking_data: List[Dict], output_path: str):
    """
    Export tracking results to CSV file.
    
    Format: [frame_id, object_id, class, x1, y1, x2, y2]
    
    Args:
        tracking_data: List of dictionaries containing tracking info per frame
        output_path: Path to save CSV file
    """
    df = pd.DataFrame(tracking_data)
    
    # Ensure correct column order
    columns = ['frame_id', 'object_id', 'class', 'x1', 'y1', 'x2', 'y2']
    df = df[columns]
    
    # Sort by frame_id and object_id
    df = df.sort_values(['frame_id', 'object_id'])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Tracking data exported to {output_path}")
    logger.info(f"Total records: {len(df)}")


def create_roi_mask(frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create Region of Interest (ROI) mask for highway scene.
    Assumes highway is in the bottom 2/3 of the frame.
    
    Args:
        frame_shape: (height, width) of frame
        
    Returns:
        Binary mask (0/255) for ROI
    """
    height, width = frame_shape
    
    # Default ROI - bottom 2/3 of frame, excluding top sky region
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if Config.ROI_VERTICES is None:
        # Create polygon for highway region
        vertices = np.array([
            [0, height],           # bottom-left
            [0, height // 3],      # top-left
            [width, height // 3],  # top-right
            [width, height]        # bottom-right
        ], np.int32)
    else:
        vertices = np.array(Config.ROI_VERTICES, np.int32)
    
    cv2.fillPoly(mask, [vertices], 255)
    
    return mask


def filter_by_roi(detections: List, mask: np.ndarray) -> List:
    """
    Filter detections based on ROI mask.
    
    Args:
        detections: List of Detection objects
        mask: ROI binary mask
        
    Returns:
        Filtered detections
    """
    filtered = []
    
    for det in detections:
        center = det.center
        
        # Check if center point is inside ROI
        if mask[center[1], center[0]] == 255:
            filtered.append(det)
    
    return filtered


class FrameProcessor:
    """Handles video I/O and frame processing pipeline."""
    
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize video processor.
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video
        """
        self.input_path = input_path
        self.output_path = output_path
        
        # Open video
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            output_path, 
            fourcc, 
            self.fps, 
            (self.width, self.height)
        )
        
        logger.info(f"Video loaded: {input_path}")
        logger.info(f"Resolution: {self.width}x{self.height}, FPS: {self.fps:.2f}, "
                   f"Frames: {self.total_frames}")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video."""
        ret, frame = self.cap.read()
        return ret, frame
    
    def write_frame(self, frame: np.ndarray):
        """Write frame to output video."""
        self.out.write(frame)
    
    def release(self):
        """Release video resources."""
        self.cap.release()
        self.out.release()
        logger.info("Video resources released")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.frame_times = []
        self.detection_times = []
        self.tracking_times = []
        self.velocity_times = []
        self.frame_count = 0
    
    def update(self, **kwargs):
        """Update timing metrics."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                getattr(self, key).append(value)
        self.frame_count += 1
    
    def get_statistics(self) -> Dict:
        """Get performance statistics."""
        stats = {}
        
        for attr in ['frame_times', 'detection_times', 'tracking_times', 'velocity_times']:
            times = getattr(self, attr)
            if times:
                stats[attr] = {
                    'mean': np.mean(times) * 1000,  # Convert to ms
                    'std': np.std(times) * 1000,
                    'min': np.min(times) * 1000,
                    'max': np.max(times) * 1000
                }
        
        stats['total_frames'] = self.frame_count
        stats['avg_fps'] = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
        
        return stats
    
    def log_statistics(self):
        """Log performance statistics."""
        stats = self.get_statistics()
        
        logger.info("=== Performance Statistics ===")
        logger.info(f"Total frames processed: {stats['total_frames']}")
        logger.info(f"Average FPS: {stats['avg_fps']:.2f}")
        
        if 'frame_times' in stats:
            logger.info(f"Frame processing time: {stats['frame_times']['mean']:.2f} ± "
                       f"{stats['frame_times']['std']:.2f} ms")
        
        if 'detection_times' in stats:
            logger.info(f"Detection time: {stats['detection_times']['mean']:.2f} ± "
                       f"{stats['detection_times']['std']:.2f} ms")
        
        if 'tracking_times' in stats:
            logger.info(f"Tracking time: {stats['tracking_times']['mean']:.2f} ± "
                       f"{stats['tracking_times']['std']:.2f} ms")
        
        if 'velocity_times' in stats:
            logger.info(f"Velocity estimation time: {stats['velocity_times']['mean']:.2f} ± "
                       f"{stats['velocity_times']['std']:.2f} ms")