"""
ByteTrack implementation with Kalman filtering for robust multi-object tracking.
Handles occlusion, re-identification, and maintains track consistency.
"""

import numpy as np
from collections import deque, OrderedDict
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass, field
from filterpy.kalman import KalmanFilter

from detector import Detection
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """Enum-like class for track states."""
    NEW = 1
    TRACKED = 2
    LOST = 3
    REMOVED = 4


class KalmanBoxTracker:
    """
    Kalman filter for bounding box tracking.
    Implements a constant velocity model for smooth trajectory estimation.
    """
    
    def __init__(self, bbox: np.ndarray, track_id: int, class_name: str, confidence: float):
        """
        Initialize Kalman filter for a new track.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            track_id: Unique identifier for this track
            class_name: Object class
            confidence: Detection confidence
        """
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        
        # Define constant velocity model
        # State: [x, y, s, r, vx, vy, vs]
        # x, y: center coordinates
        # s: scale (area)
        # r: aspect ratio
        # vx, vy, vs: velocities
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= Config.KALMAN_MEASUREMENT_NOISE
        
        # Process noise
        self.kf.P *= Config.KALMAN_PROCESS_NOISE
        
        # Initialize state
        self._initialize_state(bbox)
        
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.state = TrackState.NEW
        
        # Store history for velocity smoothing
        self.bbox_history = deque(maxlen=Config.VELOCITY_HISTORY)
        self.center_history = deque(maxlen=Config.VELOCITY_HISTORY)
        self.bbox_history.append(bbox)
        self.center_history.append(self.get_center())
    
    def _initialize_state(self, bbox: np.ndarray):
        """Initialize Kalman state from bounding box."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        s = w * h  # scale is area
        r = w / h  # aspect ratio
        
        self.kf.x[:4] = np.array([x, y, s, r])
    
    def update(self, bbox: np.ndarray):
        """Update Kalman filter with new detection."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        s = w * h
        r = w / h
        
        self.kf.update(np.array([x, y, s, r]))
        
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.state = TrackState.TRACKED
        
        # Update history
        self.bbox_history.append(self.get_bbox())
        self.center_history.append(self.get_center())
    
    def predict(self) -> np.ndarray:
        """Advance Kalman filter and return predicted bounding box."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self.get_bbox()
    
    def get_bbox(self) -> np.ndarray:
        """Convert Kalman state to bounding box."""
        x, y, s, r = self.kf.x[:4]
        w = np.sqrt(s / r)
        h = s / w
        
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        
        return np.array([x1, y1, x2, y2]).astype(int)
    
    def get_center(self) -> Tuple[int, int]:
        """Get current center position."""
        bbox = self.get_bbox()
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
    def mark_lost(self):
        """Mark track as lost."""
        self.state = TrackState.LOST
    
    def mark_removed(self):
        """Mark track as removed."""
        self.state = TrackState.REMOVED


class ByteTrack:
    """
    ByteTrack algorithm implementation with Kalman filtering.
    Handles low-confidence detections and occlusion recovery.
    """
    
    def __init__(self):
        self.track_id_count = 0
        self.active_tracks = OrderedDict()
        self.lost_tracks = OrderedDict()
        self.removed_tracks = OrderedDict()
        
        self.track_buffer = Config.TRACK_BUFFER
        self.match_threshold = Config.MATCHING_THRESHOLD
        self.high_thresh = Config.TRACK_HIGH_THRESH
        self.low_thresh = Config.TRACK_LOW_THRESH
        self.new_track_conf = Config.NEW_TRACK_CONF
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _get_iou_matrix(self, tracks: List[KalmanBoxTracker], 
                       detections: List[np.ndarray]) -> np.ndarray:
        """Compute IoU matrix between tracks and detections."""
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t, track in enumerate(tracks):
            track_bbox = track.get_bbox()
            for d, det_bbox in enumerate(detections):
                iou_matrix[t, d] = self._calculate_iou(track_bbox, det_bbox)
        return iou_matrix
    
    def _linear_assignment(self, iou_matrix: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform linear assignment using Hungarian algorithm.
        
        Returns:
            matches: List of (track_idx, det_idx) tuples
            unmatched_tracks: List of track indices
            unmatched_detections: List of detection indices
        """
        from scipy.optimize import linear_sum_assignment
        
        if iou_matrix.size == 0:
            return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))
        
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        
        matches, unmatched_tracks, unmatched_detections = [], [], []
        
        for t in range(iou_matrix.shape[0]):
            if t not in matched_indices[:, 0]:
                unmatched_tracks.append(t)
        
        for d in range(iou_matrix.shape[1]):
            if d not in matched_indices[:, 1]:
                unmatched_detections.append(d)
        
        for t, d in matched_indices:
            if iou_matrix[t, d] < threshold:
                unmatched_tracks.append(t)
                unmatched_detections.append(d)
            else:
                matches.append((t, d))
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _associate_tracks(self, tracks: List[KalmanBoxTracker], 
                         detections: List[Detection],
                         high_conf_dets: List[int],
                         low_conf_dets: List[int],
                         threshold: float) -> Tuple[Dict, List, List]:
        """Associate tracks with detections using IoU matching."""
        
        det_bboxes = [det.bbox for det in detections]
        
        # Match with high confidence detections first
        high_det_indices = high_conf_dets
        high_det_bboxes = [det_bboxes[i] for i in high_det_indices]
        
        iou_matrix = self._get_iou_matrix(tracks, high_det_bboxes)
        matches, unmatched_tracks, unmatched_high_dets = self._linear_assignment(iou_matrix, threshold)
        
        # Map back to original detection indices
        matches_dict = {}
        for track_idx, det_idx_in_high in matches:
            det_idx = high_det_indices[det_idx_in_high]
            matches_dict[track_idx] = det_idx
        
        # Now match remaining tracks with low confidence detections
        remaining_tracks = unmatched_tracks
        if remaining_tracks and low_conf_dets:
            low_det_bboxes = [det_bboxes[i] for i in low_conf_dets]
            remaining_tracks_obj = [tracks[i] for i in remaining_tracks]
            
            iou_matrix = self._get_iou_matrix(remaining_tracks_obj, low_det_bboxes)
            low_matches, low_unmatched_tracks, _ = self._linear_assignment(iou_matrix, 0.5)
            
            for track_idx_in_rem, det_idx_in_low in low_matches:
                track_idx = remaining_tracks[track_idx_in_rem]
                det_idx = low_conf_dets[det_idx_in_low]
                matches_dict[track_idx] = det_idx
        
        # Get final unmatched tracks
        unmatched_tracks = []
        for i in range(len(tracks)):
            if i not in matches_dict:
                unmatched_tracks.append(i)
        
        # Get unmatched detections
        matched_dets = set(matches_dict.values())
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        
        return matches_dict, unmatched_tracks, unmatched_dets
    
    def update(self, detections: List[Detection]) -> Dict[int, Dict]:
        """
        Main update method for ByteTrack.
        
        Args:
            detections: List of Detection objects from current frame
            
        Returns:
            Dict mapping track_id to track information
        """
        # Predict all active tracks
        for track_id, track in self.active_tracks.items():
            track.predict()
        
        # Separate detections by confidence
        high_conf_dets = []
        low_conf_dets = []
        
        for i, det in enumerate(detections):
            if det.confidence >= self.high_thresh:
                high_conf_dets.append(i)
            elif det.confidence >= self.low_thresh:
                low_conf_dets.append(i)
        
        # Get all active tracks as list
        active_track_list = list(self.active_tracks.values())
        track_id_list = list(self.active_tracks.keys())
        
        # First association step
        matches_dict, unmatched_tracks, unmatched_dets = self._associate_tracks(
            active_track_list, detections, high_conf_dets, low_conf_dets, self.match_threshold
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches_dict.items():
            track_id = track_id_list[track_idx]
            track = self.active_tracks[track_id]
            track.update(np.array(detections[det_idx].bbox))
            track.confidence = detections[det_idx].confidence
            track.class_name = detections[det_idx].class_name
        
        # Move unmatched active tracks to lost
        for track_idx in unmatched_tracks:
            track_id = track_id_list[track_idx]
            track = self.active_tracks[track_id]
            track.mark_lost()
            self.lost_tracks[track_id] = track
            del self.active_tracks[track_id]
        
        # Associate lost tracks
        if self.lost_tracks and unmatched_dets:
            lost_track_list = list(self.lost_tracks.values())
            lost_track_ids = list(self.lost_tracks.keys())
            
            # Predict lost tracks
            for track in lost_track_list:
                track.predict()
            
            remaining_dets = [detections[i] for i in unmatched_dets]
            remaining_det_bboxes = [det.bbox for det in remaining_dets]
            
            iou_matrix = self._get_iou_matrix(lost_track_list, remaining_det_bboxes)
            matches, _, _ = self._linear_assignment(iou_matrix, 0.7)
            
            # Recover matched lost tracks
            for track_idx, det_idx_in_rem in matches:
                track_id = lost_track_ids[track_idx]
                track = self.lost_tracks[track_id]
                det_idx = unmatched_dets[det_idx_in_rem]
                
                track.update(np.array(detections[det_idx].bbox))
                track.state = TrackState.TRACKED
                track.hit_streak = 1
                
                self.active_tracks[track_id] = track
                del self.lost_tracks[track_id]
        
        # Initialize new tracks from unmatched high confidence detections
        for det_idx in unmatched_dets:
            if detections[det_idx].confidence >= self.new_track_conf:
                self.track_id_count += 1
                new_track = KalmanBoxTracker(
                    np.array(detections[det_idx].bbox),
                    self.track_id_count,
                    detections[det_idx].class_name,
                    detections[det_idx].confidence
                )
                self.active_tracks[self.track_id_count] = new_track
        
        # Clean up old lost tracks
        lost_to_remove = []
        for track_id, track in self.lost_tracks.items():
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
                lost_to_remove.append(track_id)
        
        for track_id in lost_to_remove:
            self.removed_tracks[track_id] = self.lost_tracks[track_id]
            del self.lost_tracks[track_id]
        
        # Prepare output
        output = {}
        for track_id, track in self.active_tracks.items():
            output[track_id] = {
                'bbox': track.get_bbox(),
                'class_name': track.class_name,
                'confidence': track.confidence,
                'center': track.get_center(),
                'hit_streak': track.hit_streak,
                'age': track.age
            }
        
        return output
    
    def reset(self):
        """Reset tracker state."""
        self.track_id_count = 0
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()