# evaluation/2_eval_distance_lidar_REAL.py
# -*- coding: utf-8 -*-
"""
REAL Monocular Distance vs KITTI LiDAR Ground Truth (Table 5)
REAL Sensitivity Analysis (Table 6A)
REAL MiDaS Hybrid Comparison (Table 6B)
REAL Runtime Component Profiling (Table 9)

Key fixes:
  - Per-frame KITTI intrinsics injection from calib/P2
  - Fixed KITTI horizon (y ≈ 0.493 * img_h) for reliable ground-plane ranging
  - Sensitivity analysis perturbs parameters INSIDE the eval loop
  - MiDaS depth map is explicitly computed per frame for hybrid mode

No hardcoded output values. All metrics are measured.
"""

import os, sys, time, cv2, numpy as np, pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Project root for imports
# ------------------------------------------------------------------
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from adas_system import ADASSystem
    HAS_CORE = True
except ImportError as e:
    HAS_CORE = False
    print(f"[!] ADASSystem import failed: {e}")


# ------------------------------------------------------------------
# Locate KITTI
# ------------------------------------------------------------------
def find_kitti_root():
    # Explicit override takes priority, e.g.:
    #   export KITTI_ROOT=/path/to/KITTI        (Linux/macOS)
    #   set KITTI_ROOT=C:\path\to\KITTI          (Windows)
    env_root = os.environ.get("KITTI_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / "image_2").exists() and (p / "label_2").exists() and (p / "calib").exists():
            return p
        print(f"[!] KITTI_ROOT is set to '{env_root}' but 'image_2/', "
              f"'label_2/', or 'calib/' was not found there. Falling back "
              f"to default search.")

    candidates = [
        Path(r"D:\ADAS\Datasets\KITTI"),
        Path(r"C:\Users\iaelk\OneDrive - Nejran University\ADAS\Datasets\KITTI"),
        Path(r"C:\Users\iaelkhrachy\OneDrive - Nejran University\ADAS\Datasets\KITTI"),
    ]
    for c in candidates:
        if (c / "image_2").exists() and (c / "label_2").exists() and (c / "calib").exists():
            return c
    return None


KITTI_ROOT = find_kitti_root()

# KITTI horizon fraction (camera rigidly mounted)
KITTI_HORIZON_FRAC = 0.485
# KITTI camera height above road (meters)
KITTI_CAMERA_HEIGHT = 1.75

KITTI_CLASS_TO_EVAL = {
    'Car': 'Vehicles', 'Van': 'Vehicles', 'Truck': 'Vehicles',
    'Pedestrian': 'Pedestrians', 'Person_sitting': 'Pedestrians',
}


# ------------------------------------------------------------------
# KITTI loaders
# ------------------------------------------------------------------
def load_kitti_labels(frame_id):
    p = KITTI_ROOT / "label_2" / f"{frame_id}.txt"
    if not p.exists():
        return []
    out = []
    with open(p, 'r') as f:
        for line in f:
            t = line.strip().split()
            if len(t) < 15:
                continue
            cls = t[0]
            if cls not in KITTI_CLASS_TO_EVAL:
                continue
            bbox = [float(t[4]), float(t[5]), float(t[6]), float(t[7])]
            z_cam = abs(float(t[13]))
            if z_cam < 1.0 or z_cam > 50.0:
                continue
            if (bbox[2] - bbox[0]) < 15 or (bbox[3] - bbox[1]) < 15:
                continue
            out.append({
                'class': KITTI_CLASS_TO_EVAL[cls],
                'bbox': bbox,
                'distance_gt': z_cam
            })
    return out


def load_kitti_intrinsics(frame_id):
    calib_path = KITTI_ROOT / "calib" / f"{frame_id}.txt"
    if not calib_path.exists():
        return None
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith("P2:"):
                vals = [float(x) for x in line.strip().split()[1:]]
                P2 = np.array(vals).reshape(3, 4)
                return P2[:, :3].astype(np.float32)
    return None


def inject_kitti_intrinsics(adas, K, camera_height=KITTI_CAMERA_HEIGHT):
    adas.mtx = K.copy()
    adas.dist = np.zeros((5, 1), dtype=np.float32)
    adas.calibration_loaded = True
    adas.calib_width = 1242
    adas.calib_height = 375
    adas.fx_orig = float(K[0, 0])
    adas.fy_orig = float(K[1, 1])
    adas.cx_orig = float(K[0, 2])
    adas.cy_orig = float(K[1, 2])
    adas.focal_length_px = adas.fx_orig
    adas.principal_point = (adas.cx_orig, adas.cy_orig)
    adas.camera_height = camera_height
    adas.camera_pitch = 0.0
    if hasattr(adas, 'horizon_history'):
        adas.horizon_history.clear()


def bbox_iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    ab = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    return inter / (aa + ab - inter)


# ------------------------------------------------------------------
# Core evaluator with optional parameter overrides
# ------------------------------------------------------------------
def evaluate_distance_accuracy_with_perturbation(
        adas, image_files, iou_match=0.3,
        camera_height_override=None,
        horizon_frac_override=None):
    """
    Compute distance MAE/RMSE per class.
    Allows overriding camera height or horizon fraction per-frame,
    applied AFTER KITTI intrinsics injection (so they persist).
    """
    errors_by_class = {'Vehicles': [], 'Pedestrians': []}

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        K = load_kitti_intrinsics(img_path.stem)
        if K is None:
            continue
        inject_kitti_intrinsics(adas, K)

        # Apply overrides AFTER injection
        if camera_height_override is not None:
            adas.camera_height = camera_height_override
        h_frac = horizon_frac_override if horizon_frac_override is not None else KITTI_HORIZON_FRAC
        adas.current_horizon = int(img.shape[0] * h_frac)

        gts = load_kitti_labels(img_path.stem)
        if not gts:
            continue

        cars, peds, _, _ = adas.detect_objects_yolov8(img, conf_thresh=0.25)
        pool = {'Vehicles': cars, 'Pedestrians': peds}

        for gt in gts:
            cls = gt['class']
            gt_bbox = gt['bbox']
            gt_dist = gt['distance_gt']
            best_iou, best_det = 0.0, None
            for det in pool.get(cls, []):
                det_bbox = [det.x, det.y, det.x + det.width, det.y + det.height]
                i = bbox_iou(det_bbox, gt_bbox)
                if i > best_iou:
                    best_iou, best_det = i, det
            if best_iou >= iou_match and best_det and best_det.distance is not None:
                errors_by_class[cls].append(abs(best_det.distance - gt_dist))

    rows = []
    for cls, errs in errors_by_class.items():
        if not errs:
            rows.append({"Target Class": cls, "Sample Count (N)": 0,
                         "MAE (m)": "n/a", "RMSE (m)": "n/a", "Max Error (m)": "n/a"})
            continue
        errs = np.array(errs)
        rows.append({
            "Target Class": cls,
            "Sample Count (N)": int(len(errs)),
            "MAE (m)": round(float(np.mean(errs)), 3),
            "RMSE (m)": round(float(np.sqrt(np.mean(errs ** 2))), 3),
            "Max Error (m)": round(float(np.max(errs)), 3)
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Table 5 — Baseline distance accuracy
# ------------------------------------------------------------------
def evaluate_distance_accuracy(adas, image_files):
    return evaluate_distance_accuracy_with_perturbation(adas, image_files)


# ------------------------------------------------------------------
# Table 6A — Sensitivity Analysis (perturbation INSIDE eval loop)
# ------------------------------------------------------------------
def evaluate_sensitivity(adas, image_files, max_samples=50):
    sub = image_files[:max_samples]

    base_df = evaluate_distance_accuracy_with_perturbation(adas, sub)
    base_mae = float(pd.to_numeric(base_df['MAE (m)'], errors='coerce').dropna().mean())
    if np.isnan(base_mae):
        return pd.DataFrame([{"Parameter Perturbation": "Baseline",
                              "Applied Shift": "—", "MAE Increase (ΔMAE)": "n/a",
                              "Impact Severity": "n/a"}])

    rows = []
    orig_h = KITTI_CAMERA_HEIGHT
    orig_frac = KITTI_HORIZON_FRAC

    # --- Camera height ±10% ---
    h_deltas = []
    for factor in (1.10, 0.90):
        df = evaluate_distance_accuracy_with_perturbation(
            adas, sub, camera_height_override=orig_h * factor)
        mae = float(pd.to_numeric(df['MAE (m)'], errors='coerce').dropna().mean())
        if not np.isnan(mae):
            h_deltas.append(mae)
    if h_deltas:
        delta = float(np.mean(h_deltas)) - base_mae
        rows.append({
            "Parameter Perturbation": "Camera Installation Height",
            "Applied Shift": "±10%",
            "MAE Increase (ΔMAE)": f"{delta:+.3f} m",
            "Impact Severity": ("Negligible" if abs(delta) < 0.1
                                else "Moderate" if abs(delta) < 0.3 else "High")
        })

    # --- Horizon ±12 px (≈ ±2° pitch) ---
    horizon_deltas_px = 12
    p_deltas = []
    for sign in (+1, -1):
        new_frac = orig_frac + sign * (horizon_deltas_px / 375.0)
        df = evaluate_distance_accuracy_with_perturbation(
            adas, sub, horizon_frac_override=new_frac)
        mae = float(pd.to_numeric(df['MAE (m)'], errors='coerce').dropna().mean())
        if not np.isnan(mae):
            p_deltas.append(mae)
    if p_deltas:
        delta = float(np.mean(p_deltas)) - base_mae
        rows.append({
            "Parameter Perturbation": "Pitch Angle Alignment",
            "Applied Shift": "±2°",
            "MAE Increase (ΔMAE)": f"{delta:+.3f} m",
            "Impact Severity": ("Negligible" if abs(delta) < 0.1
                                else "Moderate" if abs(delta) < 0.3 else "High")
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Table 6B — MiDaS Hybrid Comparison (with explicit depth map)
# ------------------------------------------------------------------
def evaluate_midas_hybrid(adas, image_files, max_samples=20):
    sub = image_files[:max_samples]
    rows = []

    # --- Geometric only ---
    adas.depth_enabled = False
    df_geo = evaluate_distance_accuracy_with_perturbation(adas, sub)
    mae_geo = float(pd.to_numeric(df_geo['MAE (m)'], errors='coerce').dropna().mean())
    rmse_geo = float(pd.to_numeric(df_geo['RMSE (m)'], errors='coerce').dropna().mean())
    rows.append({"Estimation Mode": "Geometric Only",
                 "MAE (m)": round(mae_geo, 3) if not np.isnan(mae_geo) else "n/a",
                 "RMSE (m)": round(rmse_geo, 3) if not np.isnan(rmse_geo) else "n/a"})

    # --- Hybrid ---
    adas.depth_enabled = True
    try:
        adas.initialize_depth_model()
    except Exception as e:
        print(f"[!] MiDaS init failed: {e}")
        rows.append({"Estimation Mode": "Hybrid (Geo + MiDaS)",
                     "MAE (m)": "n/a", "RMSE (m)": "n/a"})
        return pd.DataFrame(rows)

    errors_by_class = {'Vehicles': [], 'Pedestrians': []}
    for img_path in sub:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        K = load_kitti_intrinsics(img_path.stem)
        if K is None:
            continue
        inject_kitti_intrinsics(adas, K)
        adas.current_horizon = int(img.shape[0] * KITTI_HORIZON_FRAC)

        # Force MiDaS computation for this frame
        try:
            adas.compute_depth_map(img)
        except Exception as e:
            print(f"[!] MiDaS compute failed for {img_path.stem}: {e}")
            continue

        gts = load_kitti_labels(img_path.stem)
        if not gts:
            continue
        cars, peds, _, _ = adas.detect_objects_yolov8(img, conf_thresh=0.25)
        pool = {'Vehicles': cars, 'Pedestrians': peds}

        for gt in gts:
            cls = gt['class']
            gt_bbox = gt['bbox']
            gt_dist = gt['distance_gt']
            best_iou, best_det = 0.0, None
            for det in pool.get(cls, []):
                det_bbox = [det.x, det.y, det.x + det.width, det.y + det.height]
                i = bbox_iou(det_bbox, gt_bbox)
                if i > best_iou:
                    best_iou, best_det = i, det
            if best_iou >= 0.3 and best_det and best_det.distance is not None:
                errors_by_class[cls].append(abs(best_det.distance - gt_dist))

    hybrid_errs = []
    for errs in errors_by_class.values():
        hybrid_errs.extend(errs)

    if hybrid_errs:
        arr = np.array(hybrid_errs)
        rows.append({"Estimation Mode": "Hybrid (Geo + MiDaS)",
                     "MAE (m)": round(float(np.mean(arr)), 3),
                     "RMSE (m)": round(float(np.sqrt(np.mean(arr ** 2))), 3)})
    else:
        rows.append({"Estimation Mode": "Hybrid (Geo + MiDaS)",
                     "MAE (m)": "n/a", "RMSE (m)": "n/a"})

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Table 9 — Runtime component profiling
# ------------------------------------------------------------------
def evaluate_runtime_components(adas, image_paths, n_runs=30):
    imgs = []
    for p in image_paths[:n_runs]:
        im = cv2.imread(str(p))
        if im is not None:
            imgs.append((p, im))
    if not imgs:
        return pd.DataFrame()

    first_path, first_img = imgs[0]
    K0 = load_kitti_intrinsics(first_path.stem)
    if K0 is not None:
        inject_kitti_intrinsics(adas, K0)
    adas.current_horizon = int(first_img.shape[0] * KITTI_HORIZON_FRAC)
    for _ in range(5):
        adas.process_frame(first_img)

    yolo_t, lane_t, total_t = [], [], []
    for p, img in imgs:
        K = load_kitti_intrinsics(p.stem)
        if K is not None:
            inject_kitti_intrinsics(adas, K)
        adas.current_horizon = int(img.shape[0] * KITTI_HORIZON_FRAC)

        t0 = time.perf_counter(); adas.detect_objects_yolov8(img); yolo_t.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); adas.detect_lanes(img);          lane_t.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); adas.process_frame(img);         total_t.append((time.perf_counter() - t0) * 1000)

    def s(times):
        arr = np.array(times)
        return round(float(arr.mean()), 2), round(1000.0 / float(arr.mean()), 1)

    y_ms, y_fps = s(yolo_t)
    l_ms, l_fps = s(lane_t)
    t_ms, t_fps = s(total_t)

    return pd.DataFrame([
        {"Module / Component": "YOLOv8n Object Detector",
         "Execution Latency (ms)": y_ms, "Throughput (FPS)": y_fps},
        {"Module / Component": "Classical Lane Pipeline",
         "Execution Latency (ms)": l_ms, "Throughput (FPS)": l_fps},
        {"Module / Component": "Vision-Guard Integrated Pipeline",
         "Execution Latency (ms)": t_ms, "Throughput (FPS)": t_fps},
    ])


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def run():
    print("=" * 75)
    print("REAL DISTANCE / SENSITIVITY / MIDAS / RUNTIME BENCHMARK")
    print("=" * 75)

    if not HAS_CORE or KITTI_ROOT is None:
        print("[X] Missing ADAS or KITTI dataset.")
        return

    adas = ADASSystem("config.json")
    print(f"[+] Using KITTI root: {KITTI_ROOT}")
    print(f"[i] Fixed KITTI horizon: y = {KITTI_HORIZON_FRAC:.3f} * img_h")
    print(f"[i] KITTI camera height: {KITTI_CAMERA_HEIGHT} m\n")

    image_files = sorted((KITTI_ROOT / "image_2").glob("*.png"))[:100]
    if not image_files:
        image_files = sorted((KITTI_ROOT / "image_2").glob("*.jpg"))[:100]
    print(f"[+] Loaded {len(image_files)} KITTI frames.\n")

    print("[+] TABLE 5 — Real Monocular Distance Accuracy vs LiDAR GT")
    df_t5 = evaluate_distance_accuracy(adas, image_files)
    print(df_t5.to_string(index=False), "\n")

    print("[+] TABLE 6A — Real Sensitivity Analysis")
    df_t6a = evaluate_sensitivity(adas, image_files, max_samples=30)
    print(df_t6a.to_string(index=False), "\n")

    print("[+] TABLE 6B — Real MiDaS Hybrid Comparison")
    df_t6b = evaluate_midas_hybrid(adas, image_files, max_samples=20)
    print(df_t6b.to_string(index=False), "\n")

    print("[+] TABLE 9 — Real Component Latency Profiling (MiDaS disabled)")
    adas.depth_enabled = False   # Force MiDaS off for realistic runtime
    df_t9 = evaluate_runtime_components(adas, image_files)
    print(df_t9.to_string(index=False), "\n")

    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    df_t5.to_csv(out / "table5_distance_REAL.csv", index=False)
    df_t6a.to_csv(out / "table6a_sensitivity_REAL.csv", index=False)
    df_t6b.to_csv(out / "table6b_midas_REAL.csv", index=False)
    df_t9.to_csv(out / "table9_runtime_REAL.csv", index=False)
    print(f"[+] Saved CSV outputs → {out}")


if __name__ == "__main__":
    run()