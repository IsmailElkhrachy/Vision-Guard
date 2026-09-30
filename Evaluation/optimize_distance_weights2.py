# evaluation/optimize_distance_weights.py
# -*- coding: utf-8 -*-
"""
Grid Search Optimization for Vision-Guard's Geometric Distance Formula.
FAST VERSION: YOLO runs once per frame; formula evaluated many times.
"""

import os, sys, time, cv2, itertools, numpy as np, pandas as pd, pickle
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


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
if KITTI_ROOT is None:
    print("[X] KITTI dataset not found.")
    sys.exit(1)

KITTI_TO_EVAL = {
    'Car': 'Vehicles', 'Van': 'Vehicles', 'Truck': 'Vehicles',
    'Pedestrian': 'Pedestrians', 'Person_sitting': 'Pedestrians',
}
COCO_TO_EVAL = {
    'car': 'Vehicles', 'truck': 'Vehicles', 'bus': 'Vehicles',
    'person': 'Pedestrians',
}
OBJ_HEIGHT = {'Vehicles': 1.5, 'Pedestrians': 1.7}

# Pre-optimization ("Default") configuration = Table 5C, Default column.
# The baseline MAE is now COMPUTED from these parameters (see run()),
# not hardcoded. All five values are points on the grid searched below.
DEFAULT_PARAMS = {'w1': 0.7, 'w3': 0.4, 'dist_threshold': 30,
                  'camera_height': 1.65, 'horizon_frac': 0.493}

# Value previously quoted in the manuscript; used ONLY for a consistency
# check against the live-computed baseline (never used in any result).
MANUSCRIPT_REPORTED_BASELINE_MAE = 3.878


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
            if cls not in KITTI_TO_EVAL:
                continue
            bbox = [float(t[4]), float(t[5]), float(t[6]), float(t[7])]
            z_cam = abs(float(t[13]))
            if z_cam < 1.0 or z_cam > 50.0:
                continue
            if (bbox[2] - bbox[0]) < 15 or (bbox[3] - bbox[1]) < 15:
                continue
            out.append({
                'class': KITTI_TO_EVAL[cls],
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
# PRE-COMPUTE all detections ONCE
# ------------------------------------------------------------------
def precompute_detections(model, image_files, cache_path=None):
    """
    For each frame, run YOLO and store:
      - matched detections: list of {class, bottom_y, height, dist_gt, fx, img_h}
    """
    if cache_path and Path(cache_path).exists():
        with open(cache_path, 'rb') as f:
            matched = pickle.load(f)
        print(f"[+] Loaded cached detections from {cache_path}")
        return matched

    print("[+] Pre-computing YOLO detections (this runs once)...")
    matched_samples = []

    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        K = load_kitti_intrinsics(img_path.stem)
        if K is None:
            continue
        fx = float(K[0, 0])
        img_h = img.shape[0]

        gts = load_kitti_labels(img_path.stem)
        if not gts:
            continue

        res = model(img, conf=0.25, verbose=False, device='cpu')[0]
        preds_by_cls = {}
        if res.boxes is not None:
            for b in res.boxes:
                cn = model.names[int(b.cls[0])]
                ec = COCO_TO_EVAL.get(cn)
                if ec is None:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                preds_by_cls.setdefault(ec, []).append({
                    'bbox': [x1, y1, x2, y2],
                    'bottom_y': y2,
                    'height': y2 - y1,
                })

        for gt in gts:
            cls = gt['class']
            best_iou, best_pred = 0.0, None
            for pred in preds_by_cls.get(cls, []):
                iou_val = bbox_iou(pred['bbox'], gt['bbox'])
                if iou_val > best_iou:
                    best_iou, best_pred = iou_val, pred

            if best_iou >= 0.3 and best_pred is not None:
                matched_samples.append({
                    'class': cls,
                    'bottom_y': best_pred['bottom_y'],
                    'height': best_pred['height'],
                    'dist_gt': gt['distance_gt'],
                    'fx': fx,
                    'img_h': img_h,
                })

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(image_files)}] matched samples: {len(matched_samples)}")

    print(f"[+] Pre-computation done. Total matched samples: {len(matched_samples)}")

    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(matched_samples, f)
        print(f"[+] Cached to {cache_path}")

    return matched_samples


# ------------------------------------------------------------------
# Fast formula evaluation (no YOLO)
# ------------------------------------------------------------------
def compute_distance(bottom_y, bbox_h, obj_h, img_h,
                     fx, camera_height, horizon_frac,
                     w1, w3, dist_threshold):
    v0 = int(img_h * horizon_frac)
    if bottom_y > v0:
        dist_geo = (camera_height * fx) / (bottom_y - v0)
    else:
        dist_geo = None
    if bbox_h > 0:
        dist_height = (obj_h * fx) / bbox_h
    else:
        dist_height = 50.0

    if dist_geo is None:
        return dist_height
    if dist_geo < dist_threshold:
        d = w1 * dist_geo + (1 - w1) * dist_height
    else:
        d = w3 * dist_geo + (1 - w3) * dist_height
    return max(0.5, min(d, 150.0))


def evaluate_fast(samples, params):
    errors = {'Vehicles': [], 'Pedestrians': []}
    for s in samples:
        cls = s['class']
        est = compute_distance(
            bottom_y=s['bottom_y'],
            bbox_h=s['height'],
            obj_h=OBJ_HEIGHT[cls],
            img_h=s['img_h'],
            fx=s['fx'],
            camera_height=params['camera_height'],
            horizon_frac=params['horizon_frac'],
            w1=params['w1'],
            w3=params['w3'],
            dist_threshold=params['dist_threshold'],
        )
        errors[cls].append(abs(est - s['dist_gt']))
    return errors


def summarize(errs):
    """MAE/RMSE/N for combined and per-class error dictionaries."""
    out = {}
    combined = np.array(errs['Vehicles'] + errs['Pedestrians'])
    groups = {'Combined': combined,
              'Vehicles': np.array(errs['Vehicles']),
              'Pedestrians': np.array(errs['Pedestrians'])}
    for name, arr in groups.items():
        if arr.size:
            out[name] = {'N': int(arr.size),
                         'MAE': float(arr.mean()),
                         'RMSE': float(np.sqrt((arr ** 2).mean()))}
    return out


# ------------------------------------------------------------------
def run():
    print("=" * 75)
    print("FAST GRID SEARCH — OPTIMIZING GEOMETRIC DISTANCE FORMULA")
    print("=" * 75)
    print(f"[+] KITTI root: {KITTI_ROOT}")

    model = YOLO('yolov8n.pt')

    all_images = sorted((KITTI_ROOT / "image_2").glob("*.png"))[:1000]
    calib_images = all_images[:700]
    val_images   = all_images[700:1000]
    print(f"[+] Calibration: {len(calib_images)} frames | Validation: {len(val_images)} frames\n")

    cache_dir = Path(__file__).resolve().parent / "results"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute once for calib and val
    calib_samples = precompute_detections(model, calib_images,
                                          cache_path=cache_dir / "cache_calib.pkl")
    val_samples   = precompute_detections(model, val_images,
                                          cache_path=cache_dir / "cache_val.pkl")

    # Grid search space
    grid = {
        'w1': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'w3': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'dist_threshold': [15, 20, 25, 30, 35, 40],
        'camera_height': [1.55, 1.60, 1.65, 1.70, 1.75],
        'horizon_frac': [0.470, 0.480, 0.485, 0.493, 0.500],
    }

    combos = list(itertools.product(
        grid['w1'], grid['w3'], grid['dist_threshold'],
        grid['camera_height'], grid['horizon_frac']
    ))
    print(f"[+] Grid search combinations: {len(combos)}\n")

    best_mae = float('inf')
    best_params = None
    results = []

    start = time.time()
    for i, (w1, w3, thr, ch, hf) in enumerate(combos):
        params = {'w1': w1, 'w3': w3, 'dist_threshold': thr,
                  'camera_height': ch, 'horizon_frac': hf}
        errs = evaluate_fast(calib_samples, params)
        all_errs = errs['Vehicles'] + errs['Pedestrians']
        if not all_errs:
            continue
        mae = float(np.mean(all_errs))
        rmse = float(np.sqrt(np.mean(np.array(all_errs) ** 2)))
        results.append({**params, 'MAE': mae, 'RMSE': rmse, 'N': len(all_errs)})

        if mae < best_mae:
            best_mae = mae
            best_params = params.copy()

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            print(f"    [{i+1}/{len(combos)}]  best MAE = {best_mae:.3f} m  "
                  f"({elapsed:.0f}s elapsed)")

    print(f"\n[+] Search complete in {time.time()-start:.0f}s")
    print("\n" + "=" * 75)
    print("BEST PARAMETERS (calibration):")
    print("=" * 75)
    for k, v in best_params.items():
        print(f"  {k:20s} = {v}")
    print(f"  {'MAE (calib)':20s} = {best_mae:.3f} m")

    # Validate on held-out set
    print("\n" + "=" * 75)
    print("VALIDATION ON HELD-OUT SET:")
    print("=" * 75)
    val_errs = evaluate_fast(val_samples, best_params)
    for cls, e in val_errs.items():
        if e:
            arr = np.array(e)
            print(f"  {cls:15s}: N={len(arr):4d}  "
                  f"MAE={arr.mean():.3f} m  "
                  f"RMSE={np.sqrt((arr**2).mean()):.3f} m  "
                  f"Max={arr.max():.3f} m")

    # Save top 10
    df = pd.DataFrame(results).sort_values('MAE').head(10)
    df.to_csv(cache_dir / "optimization_top10.csv", index=False)
    print(f"\n[+] Top 10 saved to {cache_dir / 'optimization_top10.csv'}")

    # ------------------------------------------------------------
    # LIVE BASELINE: default (pre-optimization) parameters evaluated on
    # the SAME calibration and validation samples as the optimized ones.
    # ------------------------------------------------------------
    default_calib = summarize(evaluate_fast(calib_samples, DEFAULT_PARAMS))
    default_val = summarize(evaluate_fast(val_samples, DEFAULT_PARAMS))
    opt_calib = summarize(evaluate_fast(calib_samples, best_params))
    opt_val = summarize(val_errs)

    print("\n" + "=" * 75)
    print("BASELINE (DEFAULT PARAMETERS, computed live):")
    print("=" * 75)
    for k, v in DEFAULT_PARAMS.items():
        print(f"  {k:20s} = {v}")
    print(f"  {'MAE (calib)':20s} = {default_calib['Combined']['MAE']:.3f} m")
    print(f"  {'MAE (validation)':20s} = {default_val['Combined']['MAE']:.3f} m")
    diff = abs(default_calib['Combined']['MAE'] - MANUSCRIPT_REPORTED_BASELINE_MAE)
    if diff < 0.0005:
        print(f"  [OK] calibration baseline matches the manuscript value "
              f"({MANUSCRIPT_REPORTED_BASELINE_MAE} m).")
    else:
        print(f"  [!] calibration baseline differs from the manuscript value "
              f"({MANUSCRIPT_REPORTED_BASELINE_MAE} m) by {diff:.3f} m -> update "
              f"Table 5C / Section 4.3.3 / cover letter with the live value.")

    print("\n" + "=" * 75)
    print("SUMMARY (all values computed from data; Combined = vehicles + pedestrians)")
    print("=" * 75)
    hdr = f"{'Set / Class':<26}{'N':>6}{'Default MAE':>13}{'Optimized MAE':>15}{'Optimized RMSE':>16}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for split, d, o in [("Calibration", default_calib, opt_calib),
                        ("Validation", default_val, opt_val)]:
        for cls in ("Combined", "Vehicles", "Pedestrians"):
            if cls in o and cls in d:
                print(f"{split + ' - ' + cls:<26}{o[cls]['N']:>6}"
                      f"{d[cls]['MAE']:>12.3f}m{o[cls]['MAE']:>14.3f}m"
                      f"{o[cls]['RMSE']:>15.3f}m")
                rows.append({'Split': split, 'Class': cls, 'N': o[cls]['N'],
                             'Default MAE (m)': round(d[cls]['MAE'], 3),
                             'Default RMSE (m)': round(d[cls]['RMSE'], 3),
                             'Optimized MAE (m)': round(o[cls]['MAE'], 3),
                             'Optimized RMSE (m)': round(o[cls]['RMSE'], 3)})
    pd.DataFrame(rows).to_csv(cache_dir / "table5c_summary.csv", index=False)

    imp_same = (default_val['Combined']['MAE'] - opt_val['Combined']['MAE']) \
        / default_val['Combined']['MAE'] * 100
    imp_calib = (default_calib['Combined']['MAE'] - opt_calib['Combined']['MAE']) \
        / default_calib['Combined']['MAE'] * 100
    imp_cross = (default_calib['Combined']['MAE'] - opt_val['Combined']['MAE']) \
        / default_calib['Combined']['MAE'] * 100
    print()
    print(f"Improvement, validation (default vs optimized, same split): {imp_same:+.1f}%")
    print(f"Improvement, calibration (default vs optimized, same split): {imp_calib:+.1f}%")
    print(f"Improvement, default-on-calibration vs optimized-on-validation "
          f"(cross-split): {imp_cross:+.1f}%")
    print(f"[+] Summary saved to {cache_dir / 'table5c_summary.csv'}")


if __name__ == "__main__":
    run()