"""
Velocity estimation module for tracked objects.
Converts pixel displacement to real-world velocity using lane width reference.
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass, field

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class VelocityEstimator:
    """
    Estimates real-world velocity of tracked objects.
    Uses lane width as reference for pixel-to-meter conversion.
    Implements exponential moving average for smooth velocity estimates.
    """
    
    fps: float
    frame_width: int
    frame_height: int
    
    def __post_init__(self):
        """Initialize velocity estimator with lane width reference."""
        self.pixels_per_meter = self._estimate_pixels_per_meter()
        self.velocities = defaultdict(lambda: deque(maxlen=Config.VELOCITY_HISTORY))
        self.previous_positions = {}
        self.smoothing_factor = Config.VELOCITY_SMOOTHING
        self.smoothed_velocities = {}
        
        logger.info(f"Velocity estimator initialized - FPS: {self.fps}, "
                   f"Pixels/meter: {self.pixels_per_meter:.2f}")
    
    def _estimate_pixels_per_meter(self) -> float:
        """
        Estimate pixel-to-meter conversion factor using lane width assumption.
        
        Assumptions:
        - Standard highway lane width = 3.7 meters
        - Approximate lane width in pixels at bottom of frame = frame_width / 4
        - This is a heuristic; can be calibrated with known distances
        """
        # Approximate lane width at bottom of frame (where vehicles are closest)
        lane_width_pixels = self.frame_width / 4.0
        
        # Convert to pixels per meter
        pixels_per_meter = lane_width_pixels / Config.LANE_WIDTH_METERS
        
        Config.PIXELS_PER_METER = pixels_per_meter
        return pixels_per_meter
    
    def _compute_pixel_velocity(self, current_pos: Tuple[int, int], 
                               track_id: int) -> Optional[float]:
        """
        Compute pixel displacement per second.
        
        Args:
            current_pos: Current (x, y) center position
            track_id: Track identifier
            
        Returns:
            Pixel velocity or None if insufficient history
        """
        if track_id in self.previous_positions:
            prev_pos = self.previous_positions[track_id]
            
            # Calculate Euclidean distance in pixels
            pixel_displacement = np.sqrt(
                (current_pos[0] - prev_pos[0]) ** 2 + 
                (current_pos[1] - prev_pos[1]) ** 2
            )
            
            # Convert to pixels per second
            pixel_velocity = pixel_displacement * self.fps
            
            return pixel_velocity
        
        return None
    
    def _pixel_to_real_velocity(self, pixel_velocity: float) -> float:
        """
        Convert pixel velocity to meters per second.
        
        Args:
            pixel_velocity: Velocity in pixels per second
            
        Returns:
            Velocity in meters per second
        """
        return pixel_velocity / self.pixels_per_meter
    
    def _apply_smoothing(self, track_id: int, raw_velocity: float) -> float:
        """
        Apply exponential moving average to velocity estimates.
        
        Args:
            track_id: Track identifier
            raw_velocity: Raw velocity estimate
            
        Returns:
            Smoothed velocity
        """
        if track_id not in self.smoothed_velocities:
            smoothed = raw_velocity
        else:
            prev_smoothed = self.smoothed_velocities[track_id]
            smoothed = (self.smoothing_factor * raw_velocity + 
                       (1 - self.smoothing_factor) * prev_smoothed)
        
        self.smoothed_velocities[track_id] = smoothed
        return smoothed
    
    def _filter_outliers(self, velocity: float) -> float:
        """
        Filter unrealistic velocities.
        
        Args:
            velocity: Velocity in m/s
            
        Returns:
            Filtered velocity (capped at reasonable values)
        """
        # Highway speed limits typically 20-40 m/s (70-144 km/h)
        MIN_VELOCITY = 0.0
        MAX_VELOCITY = 45.0  # ~160 km/h
        
        return np.clip(velocity, MIN_VELOCITY, MAX_VELOCITY)
    
    def update(self, track_id: int, center: Tuple[int, int]) -> Optional[float]:
        """
        Update velocity estimate for a track.
        
        Args:
            track_id: Track identifier
            center: Current center position
            
        Returns:
            Estimated velocity in m/s or None if not enough data
        """
        # Compute pixel velocity
        pixel_velocity = self._compute_pixel_velocity(center, track_id)
        
        if pixel_velocity is not None:
            # Convert to real velocity
            real_velocity = self._pixel_to_real_velocity(pixel_velocity)
            
            # Filter outliers
            real_velocity = self._filter_outliers(real_velocity)
            
            # Apply smoothing
            smoothed_velocity = self._apply_smoothing(track_id, real_velocity)
            
            # Store in history
            self.velocities[track_id].append(smoothed_velocity)
            
            # Update previous position
            self.previous_positions[track_id] = center
            
            return smoothed_velocity
        
        # Store current position for next frame
        self.previous_positions[track_id] = center
        return None
    
    def get_average_velocity(self, track_id: int) -> Optional[float]:
        """
        Get average velocity from history.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Average velocity or None if no history
        """
        if track_id in self.velocities and len(self.velocities[track_id]) > 0:
            return np.mean(self.velocities[track_id])
        return None
    
    def reset_track(self, track_id: int):
        """Reset velocity history for a track."""
        if track_id in self.velocities:
            del self.velocities[track_id]
        if track_id in self.previous_positions:
            del self.previous_positions[track_id]
        if track_id in self.smoothed_velocities:
            del self.smoothed_velocities[track_id]
    
    @staticmethod
    def format_velocity(velocity: float) -> str:
        """Format velocity for display."""
        if velocity is None:
            return "-- km/h"
        
        # Convert m/s to km/h for display
        kmh = velocity * 3.6
        return f"{int(kmh)} km/h"