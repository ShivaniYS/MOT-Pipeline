# Multi-Object Tracking Pipeline (Computer Vision)

## Overview
This project implements a modular multi-object tracking system for highway scenes using:

- YOLOv8 for detection
- ByteTrack for tracking
- Kalman Filter for motion prediction
- Velocity estimation using pixel-to-meter conversion

## Pipeline Architecture
Frame → Detection → Filtering → Tracking → Velocity → Visualization → CSV Export

## Modules
- detector.py → Object detection
- tracker.py → Multi-object tracking
- velocity.py → Speed estimation
- utils.py → Visualization + utilities
- main.py → Pipeline orchestrator

## Challenges Faced
Encountered compatibility issues between PyTorch serialization changes and Ultralytics model loading.

Diagnosed the issue to be caused by:
- torch.load behavior change
- version mismatches

Due to time constraints, I documented the architecture instead of forcing unstable workarounds.

## What I Learned
- How detection + tracking pipelines integrate
- Importance of environment version pinning
- Real-world debugging workflow in ML systems
