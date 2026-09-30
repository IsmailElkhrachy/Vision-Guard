# evaluation/eval_kitti_detection_REAL_v2.py
# -*- coding: utf-8 -*-
"""
Real KITTI Object Detection Evaluation (CPU-only)
Regional class focus: Car, Pedestrian, Truck (Cyclist excluded).
No hardcoded mAP values.
"""

import os, sys, time, cv2, torch, numpy as np, pandas as pd
from pathlib import Path
from ultralytics import YOLO

device = 'cpu'
torch.set_num_threads(10)

# ------------------------------------------------------------------
# AUTO-DETECT KITTI ROOT
# ------------------------------------------------------------------
def find_kitti_root():
    # Explicit override takes priority, e.g.:
    #   export KITTI_ROOT=/path/to/KITTI        (Linux/macOS)
    #   set KITTI_ROOT=C:\path\to\KITTI          (Windows)
    env_root = os.environ.get("KITTI_ROOT")
    if env_root:
        p = Path(env_root)
        if (p / "image_2").exists() and (p / "label_2").exists():
            return p
        print(f"[!] KITTI_ROOT is set to '{env_root}' but 'image_2/' or "
              f"'label_2/' was not found there. Falling back to default search.")

    candidates = [
        Path(r"D:\ADAS\Datasets\KITTI"),
        Path(r"C:\Users\iaelk\OneDrive - Nejran University\ADAS\Datasets\KITTI"),
        Path(r"C:\Users\iaelkhrachy\OneDrive - Nejran University\ADAS\Datasets\KITTI"),
    ]
    for c in candidates:
        if (c / "image_2").exists() and (c / "label_2").exists():
            return c
    for base in [Path(r"D:\ADAS"), Path(r"C:\Users\iaelk"), Path(r"C:\Users\iaelkhrachy")]:
        if base.exists():
            for p in base.rglob("image_2"):
                if p.is_dir() and (p.parent / "label_2").exists():
                    return p.parent
    return None

KITTI_ROOT = find_kitti_root()
if KITTI_ROOT is None:
    print("[X] KITTI dataset not found.")
    sys.exit(1)
print(f"[+] KITTI root: {KITTI_ROOT}")

# ------------------------------------------------------------------
# REGIONAL CLASS CONFIGURATION (Cyclist EXCLUDED)
# ------------------------------------------------------------------
KITTI_TO_EVAL = {
    'Car': 'car',
    'Van': 'car',
    'Truck': 'truck',
    'Pedestrian': 'pedestrian',
    'Person_sitting': 'pedestrian',
    # 'Cyclist': excluded - see manuscript justification
}
EVAL_CLASSES = ['car', 'pedestrian', 'truck']

COCO_TO_EVAL = {
    'car': 'car',
    'truck': 'truck',
    'bus': 'truck',
    'person': 'pedestrian',
}

# ------------------------------------------------------------------
# GROUND TRUTH LOADER
# ------------------------------------------------------------------
def load_kitti_ground_truth(frame_id):
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
                continue  # Skip Cyclist and other non-regional classes
            bbox = [float(t[4]), float(t[5]), float(t[6]), float(t[7])]
            # Filter very small or truncated objects
            if (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                continue
            out.append({'class': KITTI_TO_EVAL[cls], 'bbox': bbox})
    return out

# ------------------------------------------------------------------
# IoU + VOC AP
# ------------------------------------------------------------------
def iou(a, b):
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


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ------------------------------------------------------------------
# mAP COMPUTATION
# ------------------------------------------------------------------
def compute_map(model, image_files, iou_thresh=0.5, conf_thresh=0.25):
    all_preds = {c: [] for c in EVAL_CLASSES}
    all_gts = {c: 0 for c in EVAL_CLASSES}

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gts = load_kitti_ground_truth(img_path.stem)
        gts_by_cls = {}
        for g in gts:
            gts_by_cls.setdefault(g['class'], []).append(g)
            all_gts[g['class']] += 1

        res = model(img, conf=conf_thresh, verbose=False, device=device)[0]
        preds_by_cls = {}
        if res.boxes is not None:
            for b in res.boxes:
                cn = model.names[int(b.cls[0])]
                ec = COCO_TO_EVAL.get(cn)
                if ec is None:
                    continue
                preds_by_cls.setdefault(ec, []).append({
                    'bbox': b.xyxy[0].cpu().numpy().tolist(),
                    'conf': float(b.conf[0].cpu().numpy())
                })

        for cls in EVAL_CLASSES:
            preds = sorted(preds_by_cls.get(cls, []), key=lambda x: -x['conf'])
            gts_cls = gts_by_cls.get(cls, [])
            matched = [False] * len(gts_cls)
            for pred in preds:
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(gts_cls):
                    if matched[j]:
                        continue
                    v = iou(pred['bbox'], gt['bbox'])
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= iou_thresh and best_j >= 0:
                    matched[best_j] = True
                    all_preds[cls].append((pred['conf'], 1))
                else:
                    all_preds[cls].append((pred['conf'], 0))

    aps = {}
    for cls in EVAL_CLASSES:
        n_gt = all_gts[cls]
        preds = all_preds[cls]
        if n_gt == 0 or len(preds) == 0:
            aps[cls] = 0.0
            continue
        preds.sort(key=lambda x: -x[0])
        tp = np.array([p[1] for p in preds])
        fp = 1 - tp
        ctp = np.cumsum(tp); cfp = np.cumsum(fp)
        rec = ctp / n_gt
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        aps[cls] = voc_ap(rec, prec)

    aps['mAP'] = float(np.mean([aps[c] for c in EVAL_CLASSES]))
    return aps


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def run():
    print("=" * 75)
    print("REAL KITTI OBJECT DETECTION EVALUATION (Regional Classes: Car/Ped/Truck)")
    print("=" * 75)

    images_dir = KITTI_ROOT / "image_2"
    image_files = sorted(images_dir.glob("*.png"))[:100]
    if not image_files:
        image_files = sorted(images_dir.glob("*.jpg"))[:100]
    print(f"[+] Evaluating on {len(image_files)} real KITTI frames\n")

    rows = []
    for name in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']:
        print(f"[+] {name}")
        model = YOLO(name)
        for _ in range(5):
            _ = model(cv2.imread(str(image_files[0])), verbose=False, device=device)

        lat = []
        for p in image_files:
            img = cv2.imread(str(p))
            t0 = time.perf_counter()
            _ = model(img, verbose=False, device=device)
            lat.append((time.perf_counter() - t0) * 1000.0)
        mean_lat = float(np.mean(lat))

        ap = compute_map(model, image_files)

        rows.append({
            "YOLOv8 Variant": name.replace('.pt', ''),
            "mAP@0.5 (%)": round(ap['mAP'] * 100, 1),
            "Car AP (%)": round(ap.get('car', 0) * 100, 1),
            "Pedestrian AP (%)": round(ap.get('pedestrian', 0) * 100, 1),
            "Truck AP (%)": round(ap.get('truck', 0) * 100, 1),
            "CPU Latency (ms)": round(mean_lat, 1),
            "CPU Throughput (FPS)": round(1000.0 / mean_lat, 1),
        })
        print(f"    mAP@0.5={ap['mAP']*100:.1f}% | Car={ap.get('car',0)*100:.1f}% "
              f"| Ped={ap.get('pedestrian',0)*100:.1f}% | Truck={ap.get('truck',0)*100:.1f}% "
              f"| Latency={mean_lat:.1f}ms ({1000.0/mean_lat:.1f} FPS)")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 75)
    print(df.to_string(index=False))

    out = Path("Evaluation/results"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "table4_kitti_detection_regional.csv", index=False)
    print(f"\n[+] Saved → {out / 'table4_kitti_detection_regional.csv'}")


if __name__ == "__main__":
    run()