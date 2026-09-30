# Vision-Guard Automated Evaluation Suite

This directory contains execution scripts to reproduce and benchmark the metric values reported in **Tables 4–9** of the manuscript.

## Execution Instructions

To execute the automated evaluation pipeline on your local CPU environment:

```bash
python evaluation/eval_kitti_detection.py
python evaluation/eval_distance_lidar.py
python evaluation/eval_lane_accuracy.py