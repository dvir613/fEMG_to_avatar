# fEMG to Avatar — Project Summary

## Project Goal

Predict Facial Action Units (blendshapes) from facial surface EMG (fEMG) recorded with a DAU (Data Acquisition Unit) via BLE, and visualize them in real time on a 3D avatar in Unity. Intended for facial palsy therapy.

---

## Full Pipeline Overview

### Offline / Calibration Pipeline (runs once per participant/session)

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `experiment/experiment.py` | Records EMG from DAU via BLE socket → saves `.edf` file with annotations |
| 2 | Unity **liveCapture** project | Records facial blend shapes from camera → saves `.anim` + `.asset` files |
| 3 | `data_process/extract_Live_Capture_recorded_Data.py` | Parses `.anim` + `.asset` → outputs timestamped CSV of blend shapes |
| 4 | `data_process/classifying_ica_components.py` | Filters EDF → wavelet denoising → centers → whitens → runs PICARD ICA → saves `W.npy`, `electrode_order.npy`, `heatmap.npy` |
| 5 | `data_process/prepare_data_for_model.py` | Sliding window RMS features, face-at-rest normalization, train/test splits |
| 6 | `data_process/EMG_to_Avatar_model.py` | Trains `ImprovedEnhancedTransformNet` (16 ICA features → 31 blendshapes) → saves model `.joblib`, `scaler_X.joblib`, `scaler_Y.joblib` |

### Offline Playback (for verification, no hardware needed)

| Step | Script | What it does |
|------|--------|--------------|
| 7 | `control_avatar/send_data_to_CS.py` | Reads predicted blendshape CSV → symmetrizes to 50 blendshapes → sends via TCP (port 65432/65433) to Unity at 20 FPS |
| — | Unity **ICTFace_Avatars** project | Receives blend shapes and applies them to the avatar mesh |

### Real-Time Inference (implemented)

| Step | Script | Status |
|------|--------|--------|
| 1 | DAU hardware + Xtrodes app | Must be running and streaming on port 20001 |
| 2 | `control_avatar/send_live_to_CS.py` | Receives live EMG, processes it, runs model, sends to Unity — **done** |
| 3 | Unity **ICTFace_Avatars** | Receives and displays blend shapes in real time |

---

## Programs to Run and Order

### For calibration data collection (run simultaneously):
1. **Xtrodes BLE app** — start streaming first
2. **`experiment/experiment.py`** — launches GUI, records EMG to EDF
3. **Unity liveCapture** — record face blend shapes at the same time

### For processing and training (run sequentially after calibration):
4. `data_process/extract_Live_Capture_recorded_Data.py`
5. `data_process/classifying_ica_components.py`
6. `data_process/EMG_to_Avatar_model.py` (use saved `best_params.json` to skip hyperparameter tuning)

### For real-time visualization:
7. **Xtrodes BLE app** — start streaming
8. **Unity ICTFace_Avatars** — open and play (waits for TCP connection on port 65432)
9. **`control_avatar/send_live_to_CS.py`** — connects to DAU, loads model + scalers + atlas, runs full ICA pipeline per chunk, starts inference loop

---

## Architecture of the Real-Time Inference Script

`control_avatar/send_live_to_CS.py` is implemented. It:

1. **Loads pre-trained artifacts** from the calibration session:
   - `model.joblib` — PyTorch `ImprovedEnhancedTransformNet`
   - `scaler_X.joblib`, `scaler_Y.joblib`
   - **Atlas** (`data_process/atlas/`): 17 cluster centroids, threshold, electrode coordinates, face image grid — used for per-chunk ICA component classification

2. **Receives EMG** using `DataHandler`, accumulates into a ring buffer (2 seconds, 16 channels) on a background thread

3. **Every 50ms (20 FPS)**, runs inference replicating the full calibration pipeline:
   ```
   filter_signal(last 1s of buffer)     ← notch + bandpass
   _denoise_chunk()                     ← wavelet thresholding (db15) on full context
   center()                             ← zero-mean per channel
   whiten()                             ← sphering (SVD-based)
   picard(max_iter=50)                  ← ICA: produces W (16×16) and Y (16×T)
   _classify_rt(W, atlas)               ← interpolate |inv(W)| onto face grid,
                                           match each component to closest muscle centroid
                                           → electrode_order
   reorder Y by electrode_order
   normalize_ica_data()
   RMS over last 100ms window           ← matches window_length=0.1 used in training
   scaler_X.transform()
   model forward pass (torch.no_grad)
   scaler_Y.inverse_transform()
   fill_symetrical() → 50 blend shapes
   TCP send → Unity port 65432
   ```

---

## Key Questions Debated in This Session

### Q1: Is `send_DAUdata.py` usable for real-time inference?
**Answer:** No. It uses the old atlas/heatmap ICA approach and calls `model.predict()` as if the model is sklearn, but the current model is a PyTorch `ImprovedEnhancedTransformNet`. It needs to be replaced with a new script.

### Q2: Where does the liveCapture CSV come from?
**Answer:** `data_process/extract_Live_Capture_recorded_Data.py` already handles this. It parses the `.anim` file (blend shape curves) and the `.asset` file (recording start time), fills in missing frames, interpolates, and outputs a timestamped CSV. This is the file that `prepare_data_for_model.py` reads.

### Q3: Can you apply the calibration W matrix to new real-time chunks?
**Answer (original approach):** Yes via `W_eff = W @ whiteM_calib` — a fixed linear operator applied directly to filtered chunks, with no re-whitening.

**Current approach:** The script no longer uses a pre-saved W. Instead it re-runs the full ICA pipeline (center → whiten → PICARD) on every chunk and re-classifies components via the atlas each frame. This matches the calibration preprocessing exactly at the cost of higher per-frame compute (see Q5).

### Q4: Should you re-whiten each real-time chunk?
**Answer (original debate):** Re-whitening per chunk was considered incorrect because it projects each chunk into a different coordinate system than W was calibrated for, requiring PICARD to be re-run anyway.

**Current approach:** The script now does re-whiten and re-run PICARD on each chunk, so the W and Y are fresh each frame and component classification is done via atlas matching rather than relying on a fixed `electrode_order`. This avoids the fixed-W calibration/drift mismatch at the cost of speed.

### Q5: What about wavelet denoising in real-time?
**Original concern:** Boundary artifacts — with only 1 second of context (500 samples at 500Hz), db15 at level 5 corrupts ~38% of the window. Running denoising on a short window may make the signal worse.

**Current approach:** The script applies wavelet thresholding to the full 1-second context window as a single block (`_denoise_chunk`). Boundary artifact contamination exists but is accepted as a trade-off; if results are poor this step can be removed and calibration re-run without denoising (bandpass 35–249Hz already suppresses the main artifacts).

### Q6 (formerly Q5 note): Performance budget
Running PICARD (`max_iter=50`) + 16× `griddata` atlas classification at 20 Hz is compute-intensive. If inference exceeds 50 ms per frame, reduce `SEND_HZ` or lower `ICA_MAX_ITER` further. The `_classify_rt` griddata calls dominate; pre-computing the Delaunay triangulation could speed this up significantly.

### Q6: What window size should real-time RMS use?
**Answer:** 100ms (50 samples at 500 Hz), matching the `window_length=0.1` used in `sliding_window()` during training. Using a different window size would give the model a different feature distribution than it was trained on.

---

## Key Files and Artifacts

| File | Location | Purpose |
|------|----------|---------|
| `W.npy` | `data/participantXX/SX/` | Raw PICARD unmixing matrix (calibration only) |
| `whiteM_calib.npy` | `data/participantXX/SX/` | Calibration whitening matrix (calibration only) |
| `W_eff.npy` | `data/participantXX/SX/` | Composed operator = W @ whiteM (no longer used by real-time inference) |
| `electrode_order.npy` | `data/participantXX/SX/` | Maps ICA components to muscles (no longer used by real-time inference) |
| `*_blendshapes_ImprovedEnhancedTransformNet*.joblib` | `data/participantXX/SX/` | Trained PyTorch model |
| `scaler_X_*.joblib` | `results/` | Input feature scaler |
| `scaler_Y_*.joblib` | `results/` | Output blendshape scaler |
| `threshold.npy` | `data_process/atlas/` | Atlas classification distance threshold |
| `cluster_1.npy` … `cluster_17.npy` | `data_process/atlas/` | Muscle atlas centroids (flattened face-image arrays) |
| `side_x_coor.npy`, `side_y_coor.npy` | `data_process/atlas/` | Electrode coordinates for heatmap interpolation |
| `side.jpg` | project root | Face image used to define the interpolation grid |

---

## What Still Needs to Be Done

1. **Benchmark real-time performance** — PICARD + 16× `griddata` at 20 Hz may exceed the 50 ms budget. Profile `_classify_rt` and tune `ICA_MAX_ITER` or `SEND_HZ` as needed. Pre-computing the Delaunay triangulation (pass to `griddata` directly) is the fastest win.
2. **Evaluate boundary artifact impact of `_denoise_chunk`** — with only 500 samples of context, db15 wavelet denoising corrupts ~38% of the window. If model accuracy is poor, remove `_denoise_chunk` from `infer` and re-run calibration without wavelet denoising so the training features match.
3. **Test time synchronization** — between EDF start time and liveCapture `.asset` start time (already handled in `get_time_delta()` in `prepare_data_for_model.py`)