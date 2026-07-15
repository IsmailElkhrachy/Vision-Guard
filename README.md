# Vision‑Guard – Open‑Source ADAS for Collision Warning and Lane Departure Detection

**Vision‑Guard** is an open‑source Advanced Driver Assistance System (ADAS) that provides real‑time collision warning, lane departure detection, and metric distance estimation using a single monocular camera. It integrates YOLOv8 object detection, camera calibration, Kalman‑filtered lane tracking, optional MiDaS depth refinement, and an intuitive PyQt5 graphical user interface.

This software is designed for researchers, automotive developers, and open‑source communities. It runs on standard consumer hardware and can be used with a dashcam, a smartphone, or any USB camera.

**Key features**  
- Real‑time detection of cars, trucks, pedestrians, bicycles and traffic signs (YOLOv8)  
- Metric distance estimation using calibrated camera geometry (ground‑plane + height‑based)  
- Lane detection with Hough transform and Kalman filter smoothing  
- Audio warnings for lane departure, close vehicles (<5 m), pedestrians (<3 m), and imminent collision (<2 m)  
- Camera calibration via chessboard pattern (7×9 corners, 20 mm squares) – exports PKL, XML, TXT  
- Export detected objects to JSON or CSV (timestamp, frame, class, confidence, distance, bounding box)  
- Depth estimation option using MiDaS (monocular depth refinement)  
- Full PyQt5 GUI with video file / camera live feed, playback controls, frame‑by‑frame navigation  
- ROI (region of interest) drawing for custom masking  

## Table of Contents

1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Quick Start](#quick-start)  
4. [Camera Calibration](#camera-calibration)  
5. [Usage & GUI Overview](#usage--gui-overview)  
6. [Exporting Detections](#exporting-detections)  
7. [Configuration](#configuration)  
8. [Troubleshooting](#troubleshooting)  
9. [Citation](#citation)  
10. [License](#license)  

## Requirements

- **Operating system**: Windows 10/11, Linux (Ubuntu 20.04+), macOS (Intel/Apple Silicon)  
- **Python**: 3.10 or higher  
- **Hardware**: Any computer with a working camera (USB / built‑in). For real‑time performance a modern CPU (or a GPU with CUDA) is recommended.  
- **Optional**: NVIDIA GPU with CUDA for faster YOLOv8 inference.

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/IsmailElkhrachy/Vision-Guard.git
   cd Vision-Guard
Create a virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
Install dependencies

bash
pip install -r requirements.txt
If you don't have a requirements.txt, install manually:

bash
pip install ultralytics opencv-python torch pygame PyQt5 numpy
Download YOLO weights (optional) – the first run will automatically download yolov8n.pt. You can also place your own weights in the working directory.

Quick Start
Run the main application

bash
python main.py
Load a video file or start the camera

Click "Open Video File" to select an .mp4, .avi, or .mov file.

Click "Use Camera" to start live streaming from your default camera.

Adjust settings

Confidence threshold (slider)

Enable/disable audio warnings

Enable/disable depth estimation (MiDaS)

Observe the overlay – detected objects appear with class labels, distances, and colour‑coded bounding boxes. Lane lines are drawn in green. Warning overlays appear when a collision risk or lane departure is detected.

Camera Calibration
Accurate metric distance estimation requires calibrating your camera. The software uses a chessboard pattern (7×9 inner corners, 20 mm squares) and the rational distortion model (8 coefficients).

Steps:

Print a chessboard pattern (e.g., 7×9 corners, 20 mm squares).

Take 10–20 photos of the chessboard from different angles and distances.

In the GUI, click "Calibrate Camera", select all images.

The system computes intrinsic parameters and reprojection error.

Save the calibration as a .pkl (or .xml/.txt).

The calibration will be automatically used for distance estimation.

After calibration, you can manually adjust camera height and pitch in the Camera Geometry panel for even better accuracy.

Usage & GUI Overview
The PyQt5 interface is divided into four main areas:

Video Source – open a file or start the camera.

Playback Controls – play/pause, stop, frame‑by‑frame navigation (for video files), and speed slider.

ADAS Settings – confidence threshold, audio toggle, depth estimation toggle.

ROI Configuration – choose ROI mode (fixed, adaptive, dynamic, manual, polygon) and draw custom polygons.

Camera Calibration – calibrate or load a saved calibration.

Camera Geometry – set camera height (m) and pitch (degrees).

Results & Export – save internal results, export to CSV, and start/stop real‑time detection logging.

The video display shows lanes, bounding boxes, distance labels, and warning messages. A dark info panel in the top‑left displays lane status, curvature, offset, and object counts.

Exporting Detections
You can log all detected objects to a file for later analysis.

Click "Start Detection Export"

Choose a file name and format (JSON or CSV)

While the video/camera runs, every detected object is logged with:

timestamp, frame number, class name, confidence, estimated distance (meters), bounding box coordinates.

Click "Stop Detection Export" to close the log file.

The export is thread‑safe and does not slow down real‑time processing.

Configuration
You can customise the ADAS behaviour by editing config.json (or creating one). Example:

json
{
  "camera": {
    "focal_length": 1200,
    "camera_height": 1.5,
    "camera_pitch": 5.0,
    "processing_width": 1280,
    "processing_height": 720
  },
  "detection": {
    "confidence_threshold": 0.5,
    "yolo_model": "yolov8n.pt",
    "max_distance": 150.0
  },
  "lane": {
    "n_windows": 9,
    "margin": 100,
    "minpix": 50,
    "lane_width_pixels": 700,
    "car_position": 640
  },
  "depth": {
    "enabled": false,
    "model_type": "MiDaS_small"
  },
  "audio": {
    "enabled": true,
    "cooldown": 2.0
  }
}
Place the file in the same directory as main.py. All parameters are loaded at startup.

Troubleshooting
Problem	Possible solution
No video / camera not found	Check that the camera is not used by another application. On Linux, ensure your user has permission to access /dev/video*.
YOLO fails to load	Verify internet connection (first run downloads weights). Or download yolov8n.pt manually and place it in the working directory.
Audio warnings do not play	Install pygame and check your system’s audio output. On headless Linux, set SDL_AUDIODRIVER=dummy.
Distance estimates are inaccurate	Perform camera calibration properly. Adjust camera height and pitch in the GUI.
GUI is slow	Reduce processing resolution in config.json (e.g., 640×360). Disable depth estimation (MiDaS). Use a GPU for YOLO.
Import errors in Python	Make sure you are using Python 3.10+ and have installed all dependencies listed in requirements.txt.
For further help, please open an issue on GitHub or contact the author at iaelkhracy@nu.edu.sa.

Citation
If you use Vision‑Guard in your research, teaching, or commercial project, please cite it as:

Ismail Elkhrachy (2026). Vision‑Guard: An Open‑Source Advanced Driver Assistance System for Real‑Time Collision Warning and Lane Departure Detection (Version 1.0.0). GitHub. https://github.com/IsmailElkhrachy/Vision-Guard

BibTeX entry:

bibtex
@software{Elkhrachy_Vision-Guard_2026,
  author = {Elkhrachy, Ismail},
  title = {Vision‑Guard: An Open‑Source Advanced Driver Assistance System for Real‑Time Collision Warning and Lane Departure Detection},
  version = {1.0.0},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/IsmailElkhrachy/Vision-Guard},
  license = {MIT}
}
If you use the optional MiDaS depth module, please also cite:

R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards robust monocular depth estimation: Mixing datasets for zero‑shot cross‑dataset transfer,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 3, pp. 1623–1637, 2022.

License
This project is licensed under the MIT License – see the LICENSE file for details. You are free to use, modify, and distribute this software, provided that the original copyright notice and permission notice are retained.

Author: Ismail Elkhrachy, Department of Civil Engineering, Najran University, Saudi Arabia.
Contact: iaelkhracy@nu.edu.sa
Project link: https://github.com/IsmailElkhrachy/Vision-Guard