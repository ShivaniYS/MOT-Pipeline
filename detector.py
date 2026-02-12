"""
Detection module using YOLOv8 for object detection.
Handles model loading, inference, and post-processing.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from config import Config

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Data class for detection results."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    
    def to_list(self) -> List:
        return [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], 
                self.confidence, self.class_id, self.class_name]
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get bounding box center."""
        return ((self.bbox[0] + self.bbox[2]) // 2,
                (self.bbox[1] + self.bbox[3]) // 2)
    
    @property
    def area(self) -> int:
        """Get bounding box area."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class ObjectDetector:
    """
    YOLOv8-based object detector for highway scenes.
    Filters specific COCO classes and applies confidence threshold.
    """
    
    def __init__(self, model_path: str = Config.DETECTION_MODEL):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
        """
        self.device = Config.DEVICE
        logger.info(f"Initializing YOLO detector on {self.device}")
        
        try:
            self.model = YOLO(model_path)
            if self.device == "cuda":
                self.model.to("cuda")
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.conf_threshold = Config.CONFIDENCE_THRESHOLD
        self.iou_threshold = Config.IOU_THRESHOLD
        self.img_size = Config.IMG_SIZE
        self.target_classes = Config.TARGET_CLASSES
        self.class_mapping = Config.COCO_CLASS_MAPPING
        
        # Filtering thresholds
        self.min_area = Config.MIN_BOX_AREA
        self.min_width = Config.MIN_BOX_WIDTH
        self.min_height = Config.MIN_BOX_HEIGHT
        
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for inference.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed frame
        """
        # Basic preprocessing - resize maintaining aspect ratio
        h, w = frame.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        if scale != 1:
            frame = cv2.resize(frame, (new_w, new_h))
            
        return frame
    
    def filter_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Filter detections based on size and class.
        
        Args:
            detections: List of raw detections
            
        Returns:
            Filtered detections
        """
        filtered = []
        for det in detections:
            # Filter by class
            if det.class_id not in self.target_classes:
                continue
                
            # Filter by size
            if (det.area < self.min_area or 
                det.width < self.min_width or 
                det.height < self.min_height):
                continue
                
            # Update class name
            det.class_name = self.class_mapping[det.class_id]
            filtered.append(det)
            
        return filtered
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            frame, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )[0]
        
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detection = Detection(
                    bbox=tuple(map(int, box)),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self.class_mapping.get(cls_id, "Unknown")
                )
                detections.append(detection)
        
        # Apply filtering
        detections = self.filter_detections(detections)
        
        logger.debug(f"Detected {len(detections)} objects in frame")
        return detections
    
    def __call__(self, frame: np.ndarray) -> List[Detection]:
        """Convenience method to run detection."""
        return self.detect(frame)