# Vision‑Guard Automated Evaluation Suite

This directory contains the scripts used to generate every quantitative result reported in **Tables 4–9** (and Table 5C) of:

> Ismail Elkhrachy, *"Vision‑Guard: An Open‑Source Advanced Driver Assistance System for Real‑Time Collision Warning and Lane Departure Detection Using Monocular Vision and Deep Learning."*

All five scripts below have been verified against the manuscript. Every script is self‑contained, takes **no command‑line arguments**, and is run with a single `python <script>.py` call.

## Table‑to‑Script Mapping

| Manuscript Table | Script | What it produces |
|---|---|---|
| Table 4 — Object detection performance (YOLOv8 variants on KITTI) | `1_eval_kitti_detection_REAL_v2.py` | Per‑class AP (Car/Pedestrian/Truck) and mAP@0.5 for YOLOv8n/s/m/l, plus CPU inference latency/FPS |
| Table 5 — Distance estimation accuracy vs. LiDAR ground truth | `2_eval_distance_lidar_REAL.py` | Per‑class (Vehicles/Pedestrians) MAE, RMSE, max error, and sample count against KITTI Velodyne LiDAR ground truth |
| Table 5C — Parameter optimization (grid search) | `optimize_distance_weights2.py` | Grid search over 5,400 combinations of the geometric fusion weights (w1, w3), switching threshold, camera height, and horizon fraction; reports default‑vs‑optimized MAE on both the calibration set and a held‑out validation set |
| Table 6, Part A — Sensitivity analysis | `2_eval_distance_lidar_REAL.py` (same run as Table 5) | MAE increase from ±10% camera‑height perturbation and ±2° pitch/horizon perturbation |
| Table 6, Part B — MiDaS depth fusion comparison | `2_eval_distance_lidar_REAL.py` (same run as Table 5) | MAE/RMSE for geometry‑only vs. geometry+MiDaS hybrid fusion |
| Table 7 — Lane detection accuracy | `3_eval_lane_accuracy_REAL.py` | Frame‑level lane detection rate, accuracy, and mean lateral offset error under **Clear Daytime** and **Curved Roads** conditions |
| Table 8 — Warning system confusion matrix + trigger latency | `4_eval_warning_simulation.py` | Confusion matrix (TP/FP/FN/TN, precision, recall) for FCW, LDW, and combined multi‑hazard detection over 200 synthetic scenarios, plus warning‑dispatch trigger latency |
| Table 9 — Runtime / latency profiling | `2_eval_distance_lidar_REAL.py` (detector/lane/integrated pipeline) + `5_eval_gui_rendering.py` (GUI rendering) | Latency and throughput (FPS) for the YOLOv8n detector, the classical lane pipeline, the full integrated pipeline, and GUI frame‑rendering latency |

**Note on `5_eval_gui_rendering.py`:** this benchmarks only the `QImage`→`QPixmap` frame conversion call (100 iterations on a synthetic 1280×720 frame) — not the full Qt paint/refresh cycle. This produces the "GUI rendering (async)" row in Table 9. Make sure the standalone snippet has been removed from the top of `2_eval_distance_lidar_REAL.py` if it was pasted there during testing — it should now live only in this dedicated script, so each script's console output corresponds to exactly one manuscript table.

**Note on `3_eval_lane_accuracy_REAL.py`:** this script also contains an `evaluate_warning_system()` function for a *real‑video‑based* Table 8 (as opposed to the synthetic simulation in `4_eval_warning_simulation.py`). It is not used in the published results — no labeled real‑world warning videos were available (see the manuscript's Table 8 Note and Limitations) — but the code path remains in the script and will run automatically if you populate `evaluation/test_data/warnings/`. This is expected: the script degrades gracefully and just skips Table 8 if that folder is empty.

## Requirements

Install the same environment used to generate the published results:

```bash
pip install ultralytics==8.4.46 opencv-python==4.12.0.* torch==2.7.1 numpy pandas
```

| Component | Version used for published results |
|---|---|
| Python | 3.13.2 |
| PyTorch | 2.7.1 |
| OpenCV | 4.12.0 |
| Ultralytics (YOLOv8) | 8.4.46 |
| Hardware | Lenovo IdeaPad 3, Intel Core i7‑1255U, 16 GB RAM, **CPU‑only** (no CUDA/GPU used) |
| Operating System | Windows 11 Home (64‑bit) |

Running on different hardware or library versions may produce small numerical deviations from the published tables; this is expected and does not indicate an error.

## Dataset Setup

### KITTI Vision Benchmark (required for Tables 4, 5, 5C, 6, 9)

`1_eval_kitti_detection_REAL_v2.py`, `2_eval_distance_lidar_REAL.py`, and `optimize_distance_weights2.py` locate the KITTI dataset automatically by checking a fixed list of paths hardcoded in each script's `find_kitti_root()` function:

```
D:\ADAS\Datasets\KITTI
C:\Users\iaelk\OneDrive - Nejran University\ADAS\Datasets\KITTI
C:\Users\iaelkhrachy\OneDrive - Nejran University\ADAS\Datasets\KITTI
```

**If you are not the original author, none of these paths will exist on your machine**, and the scripts will exit with `[X] KITTI dataset not found.` To run them yourself:

1. Download the KITTI Object 3D Detection dataset (`image_2/`, `label_2/`, and `calib/` subfolders) from the [official KITTI site](https://www.cvlibs.net/datasets/kitti/).
2. Either place it at `D:\ADAS\Datasets\KITTI` (Windows), **or** edit the `candidates` list near the top of each of the three scripts above to point at your own dataset location.

Each folder must contain matching `image_2/<id>.png`, `label_2/<id>.txt`, and `calib/<id>.txt` files (standard KITTI layout).

### Custom Lane Dataset (required for Table 7)

`3_eval_lane_accuracy_REAL.py` expects video files under:

```
evaluation/test_data/lane/clear/*.mp4    — Clear Daytime
evaluation/test_data/lane/curved/*.mp4   — Curved Roads
```

The script supports four additional condition folders (`night/`, `sun/`, `rain/`, `fog/`) that will be evaluated automatically if populated, but **only `clear/` and `curved/` were used for the published Table 7** — the other four conditions are outside the current study's evaluated scope and are silently skipped if their folders are absent.

### Warning simulation (Table 8) — no dataset required

`4_eval_warning_simulation.py` generates its own synthetic scenarios programmatically (fixed random seed `SEED = 42`) and needs no external data.

## Execution Instructions

Run each script from the repository root, with `adas_system.py` present at the repo root (the scripts import it via `sys.path` insertion of the parent directory):

```bash
# Table 4 — Object detection performance
python evaluation/1_eval_kitti_detection_REAL_v2.py

# Tables 5, 6 (Parts A & B), and 9 — Distance accuracy, sensitivity, MiDaS comparison, runtime
python evaluation/2_eval_distance_lidar_REAL.py

# Table 5C — Parameter optimization (grid search, 5,400 combinations; slow — precomputes and caches YOLO detections)
python evaluation/optimize_distance_weights2.py

# Table 7 — Lane detection accuracy
python evaluation/3_eval_lane_accuracy_REAL.py

# Table 8 — Warning system simulation (200 synthetic scenarios; no dataset needed)
python evaluation/4_eval_warning_simulation.py

# Table 9 (GUI rendering row) — QImage/QPixmap conversion benchmark, no dataset needed
python evaluation/5_eval_gui_rendering.py
```

## Output Files

Each script writes CSVs to its own `results/` subdirectory next to the script:

| Script | Output files |
|---|---|
| `1_eval_kitti_detection_REAL_v2.py` | `Evaluation/results/table4_kitti_detection_regional.csv` |
| `2_eval_distance_lidar_REAL.py` | `results/table5_distance_REAL.csv`, `results/table6a_sensitivity_REAL.csv`, `results/table6b_midas_REAL.csv`, `results/table9_runtime_REAL.csv` |
| `optimize_distance_weights2.py` | `results/optimization_top10.csv`, `results/table5c_summary.csv` (default vs. optimized MAE/RMSE, combined and per class, calibration and validation), plus cached YOLO detections at `results/cache_calib.pkl` / `results/cache_val.pkl` (delete these to force a fresh detection pass) |
| `3_eval_lane_accuracy_REAL.py` | `results/table7_lane_REAL.csv` (and `results/table8_warning_REAL.csv` only if warning videos are present) |
| `4_eval_warning_simulation.py` | `results/table8_warning_simulation.csv`, `results/table8_trigger_latency.csv` |
| `5_eval_gui_rendering.py` | Console output only (`GUI rendering: X.XX ± X.XX ms`) — no CSV is written |

## Expected Output

Running the full suite should reproduce, within normal floating‑point/hardware tolerance:

- **Table 4:** Per‑class AP and mAP@0.5 for YOLOv8n/s/m/l on CPU
- **Table 5:** Vehicle MAE 3.33 m / RMSE 5.11 m; Pedestrian MAE 2.41 m
- **Table 5C:** Optimized parameters (near‑range weight 0.5, far‑range weight 0.2, threshold 40 m, camera height 1.75 m, horizon fraction 0.485); combined (vehicles + pedestrians) MAE 2.830 m on the calibration set and 2.944 m on the held‑out validation set (per‑class validation MAE: vehicles 3.067 m, pedestrians 2.237 m). The default‑configuration baseline is now **computed live** by the script and printed in a `BASELINE` block — it should match the 3.878 m reported in the manuscript; if the script prints a `[!]` mismatch warning, update the manuscript with the live value.
- **Table 6, Part A:** Camera‑height perturbation +0.210 m MAE increase (Moderate); pitch/horizon perturbation +3.344 m MAE increase (High)
- **Table 6, Part B:** Geometry‑only 2.62 m MAE vs. geometry+MiDaS 7.36 m MAE — MiDaS fusion is a documented **negative result**, not an improvement
- **Table 7:** Lane detection accuracy under Clear Daytime and Curved Roads conditions
- **Table 8** (fixed seed `42`, should match exactly):

  | Hazard Module | TP | FP | FN | TN | Precision | Recall |
  |---|---|---|---|---|---|---|
  | Forward Collision Warning (FCW) | 8,713 | 12 | 142 | 51,133 | 99.9% | 98.4% |
  | Lane Departure Warning (LDW) | 27,899 | 17 | 185 | 31,899 | 99.9% | 99.3% |
  | Combined Multi‑Hazard System | 31,935 | 23 | 252 | 27,790 | 99.9% | 99.2% |

  Trigger (dispatch) latency: **0.001 ± 0.000 ms** (mean ± std over 1,000 invocations) — this measures only the non‑blocking dispatch call overhead, not end‑to‑end audio/visual playback latency.

  These very high precision/recall values reflect the idealized nature of the simulation (small‑variance Gaussian perception noise, no ambiguous near‑threshold control events) and validate the decision logic only, not the full perception pipeline — see the manuscript's Limitations section.

- **Table 9:** YOLOv8n 44.25 ms (22.6 FPS), classical lane pipeline 6.19 ms (161.5 FPS), integrated pipeline 120.15 ms (8.3 FPS), GUI rendering (QImage/QPixmap conversion) **1.10 ± 0.40 ms** — confirmed via `5_eval_gui_rendering.py`.

If any reproduced value differs from the manuscript by more than expected floating‑point variation, please open a GitHub issue with your environment details (OS, library versions, hardware) so it can be investigated.

## Reporting Issues

If a script fails to run, produces unexpected output, or a filename/path in this README doesn't match what's actually in the repository, please open an issue at:
https://github.com/IsmailElkhrachy/Vision-Guard/issues
