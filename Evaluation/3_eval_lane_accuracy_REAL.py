# evaluation/3_eval_lane_accuracy_REAL.py
# -*- coding: utf-8 -*-
"""
REAL Lane Detection Evaluation (Table 7) and Warning System (Table 8)

Table 7 Conditions (from real videos):
    clear/    Clear Daytime      (Normal.mp4)
    curved/   Curved Roads       (Curved_roads.mp4)
    night/    Nighttime          (Nighttime.mp4)
    sun/      Direct Sun Glare   (SUN.mp4)
    rain/     Rainy Conditions   (RAIN.mp4)
    fog/      Fog / Low Vis.     (FOG.mov)

Table 8 (optional, only if you have warning videos):
    warnings/fcw_*.mp4  → collision event
    warnings/ldw_*.mp4  → lane departure event
    warnings/both_*.mp4 → both events
    warnings/safe_*.mp4 → no events
"""

import sys, time, cv2, numpy as np, pandas as pd, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from adas_system import ADASSystem
    HAS_ADAS = True
except ImportError as e:
    HAS_ADAS = False
    print(f"[!] ADASSystem import failed: {e}")


# ------------------------------------------------------------------
# Condition folder mapping (matches your videos)
# ------------------------------------------------------------------
CONDITION_FOLDERS = {
    "Clear Daytime":         "clear",
    "Curved Roads":          "curved",
    "Nighttime":             "night",
    "Direct Sun Glare":      "sun",
    "Rainy Conditions":      "rain",
    "Fog / Low Visibility":  "fog",
}

# KITTI-like offset accuracy threshold
OFFSET_ACCURACY_THRESHOLD_M = 0.10


# ------------------------------------------------------------------
# Table 7 — Real lane detection across conditions
# ------------------------------------------------------------------
def evaluate_lane_conditions(adas, root: Path, max_frames_per_video=400):
    rows = []
    for label, folder in CONDITION_FOLDERS.items():
        cond_dir = root / folder
        if not cond_dir.exists():
            print(f"[!] Missing folder: {cond_dir} — skipped.")
            continue

        vids = (list(cond_dir.glob("*.mp4")) +
                list(cond_dir.glob("*.avi")) +
                list(cond_dir.glob("*.mov")))
        if not vids:
            print(f"[!] No videos in {cond_dir} — skipped.")
            continue

        total, detected, correct = 0, 0, 0
        offsets, latencies = [], []

        for vid_path in vids:
            print(f"    → processing {vid_path.name}")
            cap = cv2.VideoCapture(str(vid_path))
            frame_count = 0
            while cap.isOpened() and frame_count < max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                t0 = time.perf_counter()
                res = adas.detect_lanes(frame)
                latencies.append((time.perf_counter() - t0) * 1000.0)

                total += 1
                if res.left_line is not None and res.right_line is not None:
                    detected += 1
                    offset = abs(res.vehicle_offset)
                    offsets.append(offset)
                    if offset <= OFFSET_ACCURACY_THRESHOLD_M:
                        correct += 1
            cap.release()
            print(f"       processed {frame_count} frames")

        if total == 0:
            continue

        rows.append({
            "Environmental Condition": label,
            "Frames": total,
            "Lane Detected Rate (%)": round(100.0 * detected / total, 1),
            "Accuracy (%)": round(100.0 * correct / total, 1),
            "Mean Offset Error (m)": round(float(np.mean(offsets)) if offsets else 0.0, 3),
            "Latency (ms)": round(float(np.mean(latencies)), 2),
            "FPS": round(1000.0 / float(np.mean(latencies)), 1),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Table 8 — Warning system (only if warnings folder exists)
# ------------------------------------------------------------------
def evaluate_warning_system(adas, videos_root: Path):
    if not videos_root.exists():
        return pd.DataFrame()

    vids = (list(videos_root.glob("*.mp4")) +
            list(videos_root.glob("*.avi")) +
            list(videos_root.glob("*.mov")))
    if not vids:
        return pd.DataFrame()

    fcw = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    ldw = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    for vid in vids:
        name = vid.name.lower()
        cap = cv2.VideoCapture(str(vid))
        triggered_fcw, triggered_ldw = False, False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            _ = adas.process_frame(frame)
            status = adas.get_warning_status()
            if status.get('collision', False):
                triggered_fcw = True
            if status.get('lane_departure', False):
                triggered_ldw = True
        cap.release()

        fcw_gt = 'fcw' in name or 'both' in name
        ldw_gt = 'ldw' in name or 'both' in name

        if fcw_gt and triggered_fcw:       fcw['tp'] += 1
        elif fcw_gt and not triggered_fcw: fcw['fn'] += 1
        elif not fcw_gt and triggered_fcw: fcw['fp'] += 1
        else:                              fcw['tn'] += 1

        if ldw_gt and triggered_ldw:       ldw['tp'] += 1
        elif ldw_gt and not triggered_ldw: ldw['fn'] += 1
        elif not ldw_gt and triggered_ldw: ldw['fp'] += 1
        else:                              ldw['tn'] += 1

    def _m(m):
        tp, fp, tn, fn = m['tp'], m['fp'], m['tn'], m['fn']
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return tp, fp, tn, fn, p, r

    rows = []
    for label, m in [("Forward Collision Warning (FCW)", fcw),
                     ("Lane Departure Warning (LDW)", ldw)]:
        tp, fp, tn, fn, p, r = _m(m)
        rows.append({
            "Hazard Module": label,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Precision (%)": round(100 * p, 1),
            "Recall (%)": round(100 * r, 1),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def run():
    print("=" * 75)
    print("REAL LANE DETECTION & WARNING SYSTEM BENCHMARK")
    print("=" * 75)

    if not HAS_ADAS:
        return

    adas = ADASSystem("config.json")
    base = Path(__file__).resolve().parent / "test_data"

    print("\n[+] TABLE 7 — Real Lane Detection Across 6 Conditions")
    df_t7 = evaluate_lane_conditions(adas, base / "lane")
    if df_t7.empty:
        print("[!] No videos found. Expected: Evaluation/test_data/lane/<condition>/")
    else:
        print(df_t7.to_string(index=False))

    print("\n[+] TABLE 8 — Real Warning System Evaluation")
    df_t8 = evaluate_warning_system(adas, base / "warnings")
    if df_t8.empty:
        print("[!] No warning videos found. This table will be reported as a simulation or future work.")
    else:
        print(df_t8.to_string(index=False))

    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    if not df_t7.empty:
        df_t7.to_csv(out / "table7_lane_REAL.csv", index=False)
    if not df_t8.empty:
        df_t8.to_csv(out / "table8_warning_REAL.csv", index=False)
    print(f"\n[+] CSV outputs → {out}")


if __name__ == "__main__":
    run()