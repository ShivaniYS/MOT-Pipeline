"""
Configuration module for highway object detection and tracking.
Contains all hyperparameters and settings for the pipeline.
"""

import torch
from pathlib import Path

class Config:
    # Paths
    INPUT_VIDEO_PATH = "challenge.mp4"
    OUTPUT_VIDEO_PATH = "output_tracking.mp4"
    OUTPUT_CSV_PATH = "tracking_output.csv"
    
    # Detection settings
    DETECTION_MODEL = "yolov8l.pt"  # Using large model for better accuracy
    CONFIDENCE_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.45
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 1280  # YOLO input size
    
    # COCO class mapping to our target classes
    COCO_CLASS_MAPPING = {
        0: "Pedestrian",    # person
        2: "Car",          # car
        3: "Motorcycle",   # motorcycle
        5: "Bus",          # bus
        7: "Truck"         # truck
    }
    TARGET_CLASSES = list(COCO_CLASS_MAPPING.keys())
    
    # Tracking settings (ByteTrack)
    TRACK_BUFFER = 30  # frames to keep lost tracks
    TRACK_HIGH_THRESH = 0.5   # high confidence detection threshold
    TRACK_LOW_THRESH = 0.1    # low confidence detection threshold
    TRACK_NMS_THRESH = 0.7    # NMS threshold for matching
    NEW_TRACK_CONF = 0.6      # confidence threshold for new tracks
    MATCHING_THRESHOLD = 0.8  # IoU threshold for matching
    
    # Kalman filter settings
    KALMAN_MEASUREMENT_NOISE = 1.0
    KALMAN_PROCESS_NOISE = 1.0
    
    # Velocity estimation
    LANE_WIDTH_METERS = 3.7  # Standard US highway lane width
    PIXELS_PER_METER = None  # Will be calculated during runtime
    VELOCITY_SMOOTHING = 0.3  # EMA smoothing factor
    VELOCITY_HISTORY = 10     # frames to average velocity
    
    # Filtering settings
    MIN_BOX_AREA = 500  # Minimum bounding box area in pixels
    MIN_BOX_WIDTH = 30  # Minimum bounding box width
    MIN_BOX_HEIGHT = 30 # Minimum bounding box height
    
    # ROI settings (optional - will auto-detect if None)
    ROI_VERTICES = None  # Will be set based on video or left as None
    
    # Visualization
    COLORS = None  # Will be generated per track ID
    FONT_SCALE = 0.6
    BOX_THICKNESS = 2
    TEXT_THICKNESS = 1
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"