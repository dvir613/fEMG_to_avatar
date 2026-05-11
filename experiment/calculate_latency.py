"""
Latency analysis for test-record EDF files.

Two complementary metrics:
  1. Absolute latency  — (pc_time - start_time) minus EDF onset.
     Positive = EDF sample counter lags real time (stream delay).
  2. Inter-annotation jitter — spacing between consecutive EDF onsets.
     Should be ~1.000 s; deviations reveal clock drift or sample-rate inaccuracy.
"""

import glob
import os
import sys
import numpy as np
import pyedflib

DEFAULT_DATA_DIR = r"C:\Users\Hila\OneDrive\מסמכים\fEMG_to_avatar\data"


def find_latest_test_edf(data_dir=DEFAULT_DATA_DIR):
    matches = glob.glob(os.path.join(data_dir, "**", "*_test.edf"), recursive=True)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_timing_annotations(edf_path):
    with pyedflib.EdfReader(edf_path) as f:
        onsets, _, labels = f.readAnnotations()

    start_time = None
    timing_checks = []

    for onset, label in zip(onsets, labels):
        label = label.strip()
        if label.startswith("Start test recording"):
            start_time = float(label.split("", 1)[1].strip())
        elif label.startswith("timing_check pc_time="):
            pc_time = float(label.split("=", 1)[1].strip())
            timing_checks.append((float(onset), pc_time))

    return start_time, timing_checks


def analyse(edf_path):
    print(f"\nFile: {edf_path}\n")

    start_time, checks = load_timing_annotations(edf_path)

    if start_time is None:
        print("ERROR: 'data_start_time' annotation not found — was this recorded in test-record mode?")
        return
    if not checks:
        print("ERROR: No 'timing_check' annotations found.")
        return

    edf_onsets = np.array([o for o, _ in checks])
    pc_times   = np.array([p for _, p in checks])

    # ── Metric 1: absolute latency ────────────────────────────────────────────
    expected_onsets = pc_times - start_time
    latencies_ms    = (expected_onsets - edf_onsets) * 1000  # positive = EDF lags wall clock

    print(f"Timing annotations : {len(checks)}")
    print(f"Recording duration : {edf_onsets[-1]:.2f} s (EDF) | "
          f"{expected_onsets[-1]:.2f} s (wall clock)")

    print("\n── Absolute latency (EDF lags wall clock) ──")
    print(f"  Mean  : {np.mean(latencies_ms):+.2f} ms")
    print(f"  Std   : {np.std(latencies_ms):.2f} ms")
    print(f"  Min   : {np.min(latencies_ms):+.2f} ms")
    print(f"  Max   : {np.max(latencies_ms):+.2f} ms")

    # ── Metric 2: inter-annotation jitter ────────────────────────────────────
    if len(edf_onsets) > 1:
        gaps_ms  = np.diff(edf_onsets) * 1000          # should all be ~1000 ms
        jitter_ms = gaps_ms - 1000.0                    # deviation from ideal 1 s

        print("\n── Inter-annotation spacing (ideal = 1000 ms) ──")
        print(f"  Mean gap : {np.mean(gaps_ms):.2f} ms")
        print(f"  Std      : {np.std(gaps_ms):.2f} ms  (jitter)")
        print(f"  Min gap  : {np.min(gaps_ms):.2f} ms  (deviation {np.min(jitter_ms):+.2f} ms)")
        print(f"  Max gap  : {np.max(gaps_ms):.2f} ms  (deviation {np.max(jitter_ms):+.2f} ms)")

    # ── Interpretation ────────────────────────────────────────────────────────
    mean_lat = np.mean(latencies_ms)
    std_lat  = np.std(latencies_ms)
    print("\n── Interpretation ──")
    if abs(mean_lat) < 50 and std_lat < 20:
        print("  OK  — stream is well-synchronized for real-time use.")
    elif abs(mean_lat) < 200 and std_lat < 50:
        print("  WARN — moderate latency; acceptable for coarse real-time feedback,")
        print("         but compensate by subtracting the mean offset in downstream code.")
    else:
        print("  FAIL — high latency or jitter; check BLE connection and packet-queue depth.")

    print()
    return latencies_ms


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        edf_path = sys.argv[1]
    else:
        edf_path = find_latest_test_edf()
        if edf_path is None:
            print(f"No *_test.edf files found under {DEFAULT_DATA_DIR}")
            print("Usage: python calculate_latency.py <path_to_test.edf>")
            sys.exit(1)
        print(f"[default] Using most recent test EDF: {edf_path}")
    analyse(edf_path)