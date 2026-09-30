# Vision‑Guard – Open‑Source ADAS for Collision Warning and Lane Departure Detection

**Vision‑Guard** is an open‑source Advanced Driver Assistance System (ADAS) that provides real‑time collision warning, lane departure detection, and metric distance estimation using a single monocular camera. It integrates YOLOv8 object detection, camera calibration, Kalman‑filtered lane tracking, an optional (disabled‑by‑default) MiDaS depth refinement module, and an intuitive PyQt5 graphical user interface.

This software is designed for researchers, automotive developers, and open‑source communities. It runs on standard consumer hardware and can be used with a dashcam, a smartphone, or any USB camera.

**Key features**
- Real‑time detection of cars, trucks, pedestrians, bicycles and traffic signs (YOLOv8)
- Metric distance estimation using calibrated camera geometry (hybrid ground‑plane + object‑height fusion)
- Lane detection with Hough transform and Kalman filter smoothing
- Audio warnings for lane departure, close vehicles (<5 m), pedestrians (<3 m), and imminent collision (<2 m)
- Camera calibration via chessboard pattern (7×9 corners, 20 mm squares) – exports PKL, XML, TXT
- Export detected objects to JSON or CSV (timestamp, frame, class, confidence, distance, bounding box)
- Optional MiDaS depth refinement module (**disabled by default** — see note below)
- Full PyQt5 GUI with video file / camera live feed, playback controls, frame‑by‑frame navigation
- ROI (region of interest) drawing for custom masking

> **Note on MiDaS depth fusion:** our published evaluation found that fusing MiDaS relative depth with the geometric distance estimator *increased* distance error under zero‑shot conditions (see the paper's Results/Limitations sections) rather than improving it. The module is kept in the codebase for future supervised‑calibration research, but `depth.enabled` defaults to `false` in `config.json`. Enable it only if you intend to re‑calibrate the depth‑to‑metric scale factor for your own camera/scene.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Camera Calibration](#camera-calibration)
5. [Usage & GUI Overview](#usage--gui-overview)
6. [Exporting Detections](#exporting-detections)
7. [Configuration](#configuration)
8. [Reproducing the Paper's Results](#reproducing-the-papers-results)
9. [Troubleshooting](#troubleshooting)
10. [Citation](#citation)
11. [License](#license)

## Requirements

- **Operating system**: Developed and tested on **Windows 11**. The codebase uses only cross‑platform libraries (PyQt5, OpenCV, PyTorch) and should run on Linux/macOS, but this has **not been independently verified** — if you run it successfully on Linux or macOS, please open an issue or PR to confirm and we'll update this section.
- **Python**: 3.10 or higher (results in the associated paper were generated with Python 3.13.2)
- **Hardware**: Any computer with a working camera (USB / built‑in). The system runs in real time on a standard laptop CPU — no GPU is required. An NVIDIA GPU with CUDA will accelerate YOLOv8 inference if available.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/IsmailElkhrachy/Vision-Guard.git
   cd Vision-Guard
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   If you don't have a `requirements.txt`, install manually:
   ```bash
   pip install ultralytics opencv-python torch pygame PyQt5 numpy
   ```

4. **Download YOLO weights (optional)** – the first run will automatically download `yolov8n.pt`. You can also place your own weights in the working directory.

## Quick Start

1. **Run the main application**
   ```bash
   python main.py
   ```

2. **Load a video file or start the camera**
   - Click **"Open Video File"** to select an `.mp4`, `.avi`, or `.mov` file.
   - Click **"Use Camera"** to start live streaming from your default camera.

3. **Adjust settings**
   - Confidence threshold (slider)
   - Enable/disable audio warnings
   - Enable/disable depth estimation (MiDaS) — off by default; see the note above before enabling it

4. **Observe the overlay** – detected objects appear with class labels, distances, and colour‑coded bounding boxes. Lane lines are drawn in green. Warning overlays appear when a collision risk or lane departure is detected.

## Camera Calibration

Accurate metric distance estimation requires calibrating your camera. The software uses a chessboard pattern (7×9 inner corners, 20 mm squares) and the rational distortion model (8 coefficients).

**Steps:**

1. Print a chessboard pattern (e.g., 7×9 corners, 20 mm squares).
2. Take 10–20 photos of the chessboard from different angles and distances.
3. In the GUI, click **"Calibrate Camera"**, select all images.
4. The system computes intrinsic parameters and reprojection error.
5. Save the calibration as a `.pkl` (or `.xml`/`.txt`).
6. The calibration will be automatically used for distance estimation.

After calibration, you can manually adjust camera height and pitch in the Camera Geometry panel for even better accuracy. Note that distance accuracy is sensitive to these two parameters in particular — see the sensitivity analysis in the paper for quantified error impact.

## Usage & GUI Overview

The PyQt5 interface is divided into several areas:

- **Video Source** – open a file or start the camera.
- **Playback Controls** – play/pause, stop, frame‑by‑frame navigation (for video files), and speed slider.
- **ADAS Settings** – confidence threshold, audio toggle, depth estimation toggle.
- **ROI Configuration** – choose ROI mode (fixed, adaptive, dynamic, manual, polygon) and draw custom polygons.
- **Camera Calibration** – calibrate or load a saved calibration.
- **Camera Geometry** – set camera height (m) and pitch (degrees).
- **Results & Export** – save internal results, export to CSV, and start/stop real‑time detection logging.

The video display shows lanes, bounding boxes, distance labels, and warning messages. A dark info panel in the top‑left displays lane status, curvature, offset, and object counts.

## Exporting Detections

You can log all detected objects to a file for later analysis.

1. Click **"Start Detection Export"**.
2. Choose a file name and format (JSON or CSV).
3. While the video/camera runs, every detected object is logged with: timestamp, frame number, class name, confidence, estimated distance (meters), bounding box coordinates.
4. Click **"Stop Detection Export"** to close the log file.

The export is thread‑safe and does not slow down real‑time processing.

## Configuration

You can customise the ADAS behaviour by editing `config.json` (or creating one). Example:

```json
{
  "camera": {
    "focal_length": 1200,
    "camera_height": 1.75,
    "camera_pitch": 0.0,
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
    "cooldown": 2.0,
    "sound_folder": "./sounds/"
  }
}
```

Place the file in the same directory as `main.py`. All parameters are loaded at startup; any field you omit falls back to the corresponding default in `adas_system.py`.

> ⚠️ **Cross‑platform note:** if `audio.sound_folder` is not set in `config.json`, the software currently falls back to a Windows‑specific default path. Always set `sound_folder` explicitly (e.g., `"./sounds/"`) if you're running on Linux or macOS, or if you move the project to a different machine.

The default distance‑fusion weights baked into `adas_system.py` (near‑range weight 0.5, far‑range weight 0.2, switching threshold 40 m, default horizon fraction 0.485) are the parameters obtained from the grid‑search optimization described in the paper (Section 4.3.3 / Table 5C), not arbitrary defaults — changing them will change reported accuracy.

## Reproducing the Paper's Results

All quantitative results in the associated publication (Tables 4–9) were generated with the scripts in `/evaluation`. See [`evaluation/README.md`](evaluation/README.md) for exact commands, dataset requirements, and the software/hardware environment used.

Summary of the environment used to generate the paper's numbers:

| Component | Version |
|---|---|
| Python | 3.13.2 |
| PyTorch | 2.7.1 |
| OpenCV | 4.12.0 |
| Ultralytics (YOLOv8) | 8.4.46 |
| PyQt5 | 5.15.2 |
| Hardware | Lenovo IdeaPad 3, Intel Core i7‑1255U, CPU‑only (no GPU used) |

If you reproduce results on different library versions or hardware, small numerical differences from the published tables are expected — please open an issue if you see a discrepancy beyond normal floating‑point/hardware variation.

## Troubleshooting

| Problem | Possible solution |
|---|---|
| No video / camera not found | Check that the camera is not used by another application. On Linux, ensure your user has permission to access `/dev/video*`. |
| YOLO fails to load | Verify internet connection (first run downloads weights). Or download `yolov8n.pt` manually and place it in the working directory. |
| Audio warnings do not play | Install `pygame` and check your system's audio output. On headless Linux, set `SDL_AUDIODRIVER=dummy`. If you're on Linux/macOS, also confirm `audio.sound_folder` is set in `config.json` (see Configuration above) — without it, the software defaults to a Windows‑only path. |
| Distance estimates are inaccurate | Perform camera calibration properly. Adjust camera height and pitch in the GUI — these are the two parameters the system is most sensitive to (see the paper's sensitivity analysis). |
| GUI is slow | Reduce processing resolution in `config.json` (e.g., 640×360). Keep depth estimation (MiDaS) disabled — it is off by default and was not found to help accuracy. Use a GPU for YOLO if available. |
| Import errors in Python | Make sure you are using Python 3.10+ and have installed all dependencies listed in `requirements.txt`. |

For further help, please open an issue on GitHub or contact the author at **iaelkhrachy@nu.edu.sa**.

## Citation

If you use Vision‑Guard in your research, teaching, or commercial project, please cite it as:

Ismail Elkhrachy (2026). *Vision‑Guard: An Open‑Source Advanced Driver Assistance System for Real‑Time Collision Warning and Lane Departure Detection* (Version 1.0.0). GitHub. https://github.com/IsmailElkhrachy/Vision-Guard

BibTeX entry:

```bibtex
@software{Elkhrachy_Vision-Guard_2026,
  author = {Elkhrachy, Ismail},
  title = {Vision‑Guard: An Open‑Source Advanced Driver Assistance System for Real‑Time Collision Warning and Lane Departure Detection},
  version = {1.0.0},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/IsmailElkhrachy/Vision-Guard},
  license = {MIT}
}
```

If you use the optional MiDaS depth module, please also cite:

R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, "Towards robust monocular depth estimation: Mixing datasets for zero‑shot cross‑dataset transfer," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 3, pp. 1623–1637, 2022.

## License

This project is licensed under the MIT License – see the `LICENSE` file for details. You are free to use, modify, and distribute this software, provided that the original copyright notice and permission notice are retained.

**Author:** Ismail Elkhrachy, Department of Civil Engineering, Najran University, Saudi Arabia.
**Contact:** iaelkhrachy@nu.edu.sa
**Project link:** https://github.com/IsmailElkhrachy/Vision-Guard
