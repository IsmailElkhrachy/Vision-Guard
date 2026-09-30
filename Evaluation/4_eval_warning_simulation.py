"""
4_eval_warning_simulation.py

Controlled synthetic simulation for the Vision-Guard warning subsystem.

Produces Table 8 (confusion matrix + precision/recall) and the trigger
latency statistics reported in Section 4.5 of the manuscript.

Scenario composition (200 total, 300 frames each, 30 FPS, 10 s per
scenario, 60,000 frames total):
    - 50 Forward Collision Warning (FCW) scenarios
    - 50 Lane Departure Warning (LDW) scenarios
    - 50 Dual-event scenarios (FCW and LDW within the same trajectory)
    - 50 Control scenarios (normal driving, no expected warnings)

Warning logic (identical thresholds to Vision-Guard defaults):
    - FCW triggers when estimated headway < 2.0 m
    - LDW triggers when estimated |lateral offset| > 0.30 m
    - Two consecutive frames are required to confirm a warning

Ground truth is read directly from the simulator's internal state.
Sensor noise is injected to emulate real ADAS perception.

Outputs (written to ./results/):
    table8_warning_simulation.csv   - confusion matrix per hazard module
    table8_trigger_latency.csv      - dispatch latency samples
"""

import os
import csv
import time
import numpy as np

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SEED = 42
FPS = 30
DURATION_S = 10.0
FRAMES_PER_SCENARIO = int(round(FPS * DURATION_S))          # 300

N_FCW = 50
N_LDW = 50
N_DUAL = 50
N_CONTROL = 50
N_SCENARIOS = N_FCW + N_LDW + N_DUAL + N_CONTROL           # 200
N_FRAMES_TOTAL = N_SCENARIOS * FRAMES_PER_SCENARIO          # 60,000

# Warning thresholds (identical to Vision-Guard defaults)
FCW_THRESHOLD_M = 2.0
LDW_THRESHOLD_M = 0.30

# Sensor noise (1-sigma) for the estimated quantities
SIGMA_HEADWAY_M = 0.15
SIGMA_OFFSET_M = 0.02

# Temporal confirmation (consecutive frames above threshold)
WARN_CONFIRM_FRAMES = 2

# Evaluation window around ground-truth onset (documented, not used
# for frame-level confusion since each frame is evaluated independently)
TP_WINDOW_MS = 100.0
TP_WINDOW_FRAMES = int(round(TP_WINDOW_MS / 1000.0 * FPS))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------
# Non-blocking warning dispatch (emulated)
# ------------------------------------------------------------------
_warning_buffer = []
_MAX_BUFFER = 10000


def _emit_warning(kind, frame_idx, t_sec):
    """Emulate a non-blocking audio/visual warning dispatch."""
    msg = f"[WARN] {kind} @ frame={frame_idx}, t={t_sec:.3f}s"
    entry = (kind, frame_idx, t_sec, msg, time.perf_counter_ns())
    _warning_buffer.append(entry)
    if len(_warning_buffer) > _MAX_BUFFER:
        _warning_buffer.clear()


# ------------------------------------------------------------------
# Scenario generators
# ------------------------------------------------------------------
def _time_array():
    return np.arange(FRAMES_PER_SCENARIO) / FPS


def generate_fcw_scenario():
    """Ego approaches a lead vehicle until headway < 2.0 m."""
    t = _time_array()
    h0 = rng.uniform(15.0, 30.0)
    rate = rng.uniform(2.0, 4.0)                    # m/s closing speed
    headway = np.maximum(h0 - rate * t, 0.5)
    offset = 0.05 * np.sin(2 * np.pi * 0.1 * t) \
             + rng.normal(0, 0.01, FRAMES_PER_SCENARIO)
    offset = np.clip(offset, -0.10, 0.10)
    return headway, offset


def generate_ldw_scenario():
    """Ego drifts laterally until |offset| > 0.30 m."""
    t = _time_array()
    base_h = rng.uniform(15.0, 30.0)
    headway = np.clip(base_h + rng.normal(0, 0.3, FRAMES_PER_SCENARIO), 5.0, None)

    sign = 1.0 if rng.random() < 0.5 else -1.0
    o0 = sign * rng.uniform(0.0, 0.10)
    target = sign * rng.uniform(0.6, 1.2)
    tau = rng.uniform(1.0, 2.5)
    offset = target * (1 - np.exp(-t / tau)) + o0 \
             + rng.normal(0, 0.01, FRAMES_PER_SCENARIO)
    return headway, offset


def generate_dual_scenario():
    """Both FCW and LDW events occur in the same trajectory."""
    t = _time_array()
    h0 = rng.uniform(15.0, 30.0)
    rate = rng.uniform(2.0, 4.0)
    headway = np.maximum(h0 - rate * t, 0.5)

    sign = 1.0 if rng.random() < 0.5 else -1.0
    o0 = sign * rng.uniform(0.0, 0.10)
    target = sign * rng.uniform(0.6, 1.2)
    tau = rng.uniform(1.0, 2.5)
    offset = target * (1 - np.exp(-t / tau)) + o0 \
             + rng.normal(0, 0.01, FRAMES_PER_SCENARIO)
    return headway, offset


def generate_control_scenario():
    """Normal driving: no warnings expected."""
    t = _time_array()
    base_h = rng.uniform(15.0, 30.0)
    headway = np.clip(base_h + rng.normal(0, 0.3, FRAMES_PER_SCENARIO), 5.0, None)
    offset = 0.08 * np.sin(2 * np.pi * 0.2 * t) \
             + rng.normal(0, 0.01, FRAMES_PER_SCENARIO)
    offset = np.clip(offset, -0.15, 0.15)
    return headway, offset


# ------------------------------------------------------------------
# Warning logic
# ------------------------------------------------------------------
def _confirm_consecutive(raw, k):
    """True only after k consecutive True frames ending at current index."""
    out = np.zeros(len(raw), dtype=bool)
    count = 0
    for i in range(len(raw)):
        count = count + 1 if raw[i] else 0
        if count >= k:
            out[i] = True
    return out


def warning_logic(measured_headway, measured_offset):
    """Apply Vision-Guard's warning thresholds and temporal confirmation."""
    raw_fcw = measured_headway < FCW_THRESHOLD_M
    raw_ldw = np.abs(measured_offset) > LDW_THRESHOLD_M

    fcw_warn = _confirm_consecutive(raw_fcw, WARN_CONFIRM_FRAMES)
    ldw_warn = _confirm_consecutive(raw_ldw, WARN_CONFIRM_FRAMES)

    latencies_ms = []
    t_arr = np.arange(FRAMES_PER_SCENARIO) / FPS
    for i in range(FRAMES_PER_SCENARIO):
        if fcw_warn[i] or ldw_warn[i]:
            t0 = time.perf_counter_ns()
            if fcw_warn[i]:
                _emit_warning("FCW", i, t_arr[i])
            if ldw_warn[i]:
                _emit_warning("LDW", i, t_arr[i])
            t1 = time.perf_counter_ns()
            latencies_ms.append((t1 - t0) / 1e6)

    return fcw_warn, ldw_warn, latencies_ms


# ------------------------------------------------------------------
# Confusion-matrix helpers
# ------------------------------------------------------------------
def _update_confusion(acc, gt, warn):
    acc["TP"] += int(np.sum(gt & warn))
    acc["FP"] += int(np.sum(~gt & warn))
    acc["FN"] += int(np.sum(gt & ~warn))
    acc["TN"] += int(np.sum(~gt & ~warn))


def _metrics(acc):
    tp, fp, fn = acc["TP"], acc["FP"], acc["FN"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 78)
    print("REAL WARNING SYSTEM SIMULATION - Vision-Guard")
    print("=" * 78)
    print(f"Scenarios          : {N_SCENARIOS} "
          f"({N_FCW} FCW, {N_LDW} LDW, {N_DUAL} dual, {N_CONTROL} control)")
    print(f"Frames per scenario: {FRAMES_PER_SCENARIO} "
          f"({DURATION_S:.0f} s @ {FPS} FPS)")
    print(f"Total frames       : {N_FRAMES_TOTAL}")
    print(f"Thresholds         : FCW < {FCW_THRESHOLD_M} m, "
          f"LDW > {LDW_THRESHOLD_M} m")
    print(f"Confirmation window: {WARN_CONFIRM_FRAMES} frames")
    print()

    confusion = {
        "fcw":      {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
        "ldw":      {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
        "combined": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
    }

    all_latencies_ms = []

    scenario_types = (["fcw"] * N_FCW
                      + ["ldw"] * N_LDW
                      + ["dual"] * N_DUAL
                      + ["control"] * N_CONTROL)
    rng.shuffle(scenario_types)

    for scen_type in scenario_types:
        if scen_type == "fcw":
            true_h, true_o = generate_fcw_scenario()
        elif scen_type == "ldw":
            true_h, true_o = generate_ldw_scenario()
        elif scen_type == "dual":
            true_h, true_o = generate_dual_scenario()
        else:
            true_h, true_o = generate_control_scenario()

        # Ground truth from simulator internal state
        gt_fcw = true_h < FCW_THRESHOLD_M
        gt_ldw = np.abs(true_o) > LDW_THRESHOLD_M
        gt_combined = gt_fcw | gt_ldw

        # Noisy sensor measurements
        meas_h = true_h + rng.normal(0, SIGMA_HEADWAY_M, FRAMES_PER_SCENARIO)
        meas_o = true_o + rng.normal(0, SIGMA_OFFSET_M, FRAMES_PER_SCENARIO)

        # Warning logic
        fcw_warn, ldw_warn, latencies = warning_logic(meas_h, meas_o)
        combined_warn = fcw_warn | ldw_warn
        all_latencies_ms.extend(latencies)

        # Confusion matrices
        _update_confusion(confusion["fcw"], gt_fcw, fcw_warn)
        _update_confusion(confusion["ldw"], gt_ldw, ldw_warn)
        _update_confusion(confusion["combined"], gt_combined, combined_warn)

    # Build report rows
    rows = []
    for key, label in [("fcw", "Forward Collision Warning (FCW)"),
                       ("ldw", "Lane Departure Warning (LDW)"),
                       ("combined", "Combined Multi-Hazard System")]:
        acc = confusion[key]
        p, r = _metrics(acc)
        rows.append({
            "Hazard Module": label,
            "TP": acc["TP"], "FP": acc["FP"],
            "FN": acc["FN"], "TN": acc["TN"],
            "Precision (%)": round(p * 100, 1),
            "Recall (%)": round(r * 100, 1),
        })

    print("[+] TABLE 8 - Real Warning System Evaluation (simulation)")
    header = (f"{'Hazard Module':<34}{'TP':>7}{'FP':>7}"
              f"{'FN':>7}{'TN':>7}{'Precision':>12}{'Recall':>10}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['Hazard Module']:<34}"
              f"{row['TP']:>7}{row['FP']:>7}{row['FN']:>7}{row['TN']:>7}"
              f"{row['Precision (%)']:>11.1f}%{row['Recall (%)']:>9.1f}%")
    print()

    # Trigger latency
    lat = np.asarray(all_latencies_ms, dtype=float)
    if lat.size == 0:
        lat_mean, lat_std, n_lat = 0.0, 0.0, 0
    else:
        if lat.size > 1000:
            idx = rng.choice(lat.size, size=1000, replace=False)
            lat_report = lat[idx]
        else:
            lat_report = lat
        lat_mean = float(np.mean(lat_report))
        lat_std = float(np.std(lat_report))
        n_lat = lat_report.size

    print(f"[+] Trigger latency: {lat_mean:.3f} +/- {lat_std:.3f} ms "
          f"(mean +/- std over {n_lat} invocations)")
    print()

    # Save CSVs
    csv_path = os.path.join(RESULTS_DIR, "table8_warning_simulation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Hazard Module", "TP", "FP", "FN", "TN",
                           "Precision (%)", "Recall (%)"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[+] Saved -> {csv_path}")

    lat_path = os.path.join(RESULTS_DIR, "table8_trigger_latency.csv")
    with open(lat_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["latency_ms"])
        for v in all_latencies_ms:
            writer.writerow([f"{v:.6f}"])
    print(f"[+] Saved -> {lat_path}")


if __name__ == "__main__":
    main()