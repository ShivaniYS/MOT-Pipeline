#!/usr/bin/env python3
"""
Main entry point for multi-class object detection and tracking in highway driving.
Implements complete pipeline with YOLOv8, ByteTrack, and velocity estimation.
"""
import numpy as np

import argparse
import time
import logging
from pathlib import Path
import sys
from collections import defaultdict

# Import local modules
from config import Config
from detector import ObjectDetector
from tracker import ByteTrack
from velocity import VelocityEstimator
from utils import (
    setup_logging, draw_detections, export_to_csv,
    create_roi_mask, filter_by_roi, FrameProcessor,
    PerformanceMonitor
)

logger = logging.getLogger(__name__)


class HighwayTrackingPipeline:
    """
    Complete pipeline for highway object detection, tracking, and velocity estimation.
    """
    
    def __init__(self, input_path: str, output_path: str, csv_path: str):
        """
        Initialize the tracking pipeline.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            csv_path: Path to output CSV
        """
        self.input_path = input_path
        self.output_path = output_path
        self.csv_path = csv_path
        
        # Initialize components
        logger.info("Initializing detection module...")
        self.detector = ObjectDetector()
        
        logger.info("Initializing tracker module...")
        self.tracker = ByteTrack()
        
        # Initialize video processor
        self.video_processor = FrameProcessor(input_path, output_path)
        
        # Initialize velocity estimator
        self.velocity_estimator = VelocityEstimator(
            fps=self.video_processor.fps,
            frame_width=self.video_processor.width,
            frame_height=self.video_processor.height
        )
        
        # Initialize ROI mask
        self.roi_mask = create_roi_mask(
            (self.video_processor.height, self.video_processor.width)
        )
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Data collection for CSV
        self.tracking_data = []
        
        logger.info("Pipeline initialized successfully")
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input BGR frame
            frame_id: Current frame number
            
        Returns:
            Annotated frame
        """
        frame_start_time = time.time()
        
        # 1. Detection
        det_start = time.time()
        detections = self.detector.detect(frame)
        det_time = time.time() - det_start
        
        # Apply ROI filtering
        if self.roi_mask is not None:
            detections = filter_by_roi(detections, self.roi_mask)
        
        # 2. Tracking
        track_start = time.time()
        tracks = self.tracker.update(detections)
        track_time = time.time() - track_start
        
        # 3. Velocity estimation
        vel_start = time.time()
        velocities = {}
        for track_id, track_info in tracks.items():
            center = track_info['center']
            velocity = self.velocity_estimator.update(track_id, center)
            velocities[track_id] = velocity
            
            # Collect data for CSV
            bbox = track_info['bbox']
            self.tracking_data.append({
                'frame_id': frame_id,
                'object_id': track_id,
                'class': track_info['class_name'],
                'x1': bbox[0],
                'y1': bbox[1],
                'x2': bbox[2],
                'y2': bbox[3]
            })
        vel_time = time.time() - vel_start
        
        # 4. Visualization
        display_frame = draw_detections(frame, tracks, velocities)
        
        # Update performance metrics
        frame_time = time.time() - frame_start_time
        self.performance_monitor.update(
            frame_times=frame_time,
            detection_times=det_time,
            tracking_times=track_time,
            velocity_times=vel_time
        )
        
        # Log progress
        if frame_id % 100 == 0:
            logger.info(f"Processed frame {frame_id}/{self.video_processor.total_frames}")
        
        return display_frame
    
    def run(self):
        """Run the complete tracking pipeline."""
        logger.info("Starting tracking pipeline...")
        
        frame_id = 0
        
        try:
            while True:
                ret, frame = self.video_processor.read_frame()
                
                if not ret:
                    break
                
                frame_id += 1
                
                # Process frame
                display_frame = self.process_frame(frame, frame_id)
                
                # Write output
                self.video_processor.write_frame(display_frame)
        
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            raise
        
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and save results."""
        logger.info("Cleaning up pipeline...")
        
        # Release video resources
        self.video_processor.release()
        
        # Export tracking data
        if self.tracking_data:
            export_to_csv(self.tracking_data, self.csv_path)
        
        # Log performance statistics
        self.performance_monitor.log_statistics()
        
        logger.info("Pipeline completed successfully")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Class Object Detection and Tracking in Highway Driving"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default=Config.INPUT_VIDEO_PATH,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=Config.OUTPUT_VIDEO_PATH,
        help="Path to output video file"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default=Config.OUTPUT_CSV_PATH,
        help="Path to output CSV file"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=Config.CONFIDENCE_THRESHOLD,
        help="Detection confidence threshold"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=Config.DEVICE,
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=Config.DETECTION_MODEL,
        help="YOLO model path"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Setup logging
    setup_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    # Update configuration
    Config.CONFIDENCE_THRESHOLD = args.conf
    Config.DEVICE = args.device
    Config.DETECTION_MODEL = args.model
    
    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    try:
        pipeline = HighwayTrackingPipeline(
            input_path=args.input,
            output_path=args.output,
            csv_path=args.csv
        )
        
        pipeline.run()
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()