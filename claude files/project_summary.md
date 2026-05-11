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

### Real-Time Inference (goal — partially implemented)

| Step | Script | Status |
|------|--------|--------|
| 1 | DAU hardware + Xtrodes app | Must be running and streaming on port 20001 |
| 2 | `control_avatar/send_live_to_CS.py` *(to be written)* | Receives live EMG, processes it, runs model, sends to Unity |
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
9. **`control_avatar/send_live_to_CS.py`** *(to be written)* — connects to DAU, loads W_eff + model + scalers, starts inference loop

---

## Architecture of the Real-Time Inference Script

The `send_live_to_CS.py` script must:

1. **Load pre-trained artifacts** from the calibration session:
   - `W_eff.npy` — composed unmixing matrix (= `W @ whiteM_calib`)
   - `electrode_order.npy` — component-to-muscle mapping
   - `model.joblib` — PyTorch `ImprovedEnhancedTransformNet`
   - `scaler_X.joblib`, `scaler_Y.joblib`

2. **Receive EMG** using `DataHandler` (same as `experiment.py`), accumulate into a ring buffer (~2 seconds, 16 channels)

3. **Every 50ms (20 FPS)**, run inference:
   ```
   filter_signal(last 1s of buffer)     ← notch + bandpass
   W_eff @ filtered_chunk               ← apply fixed unmixing operator
   reorder by electrode_order
   normalize_ica_data()
   RMS over last 100ms window           ← matches training window size
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
**Answer:** Yes, but not by applying `W` alone. The PICARD ICA was run on data that was already whitened with a whitening matrix `whiteM_calib` computed from the calibration recording. So `W` was designed to unmix whitened data. The correct operator for raw data is:
```
W_eff = W @ whiteM_calib
```
This `W_eff` is the actual inverse of the physical mixing matrix A (determined by electrode geometry, which is fixed). It should be computed and saved during calibration, and applied directly to filtered real-time chunks.

### Q4: Should you re-whiten each real-time chunk?
**Answer:** No — this is wrong. If you compute a new whitening matrix per chunk, you project the chunk into a different coordinate system than W was calibrated for. Applying W to a differently-whitened chunk gives components that no longer correspond to the same muscles. You would need to re-run PICARD (minutes of computation) to get a valid W for each new whitening. The whole point of calibration is that `W_eff` is a fixed linear operator (the physical inverse mixing matrix). Apply it directly.

### Q5: What about wavelet denoising in real-time?
**Answer:** Unavoidable mismatch. During calibration, wavelet denoising was applied before computing `whiteM_calib` and `W`. In real-time, you cannot do wavelet denoising (it is a batch operation and too slow). Practical compromise:
- The bandpass filter (35–249 Hz) removes most of what wavelet denoising targets
- `normalize_ica_data()` provides robustness to residual amplitude differences
- Real-time ICA quality will be slightly lower than calibration quality, which is acceptable

### Q6: What window size should real-time RMS use?
**Answer:** 100ms (50 samples at 500 Hz), matching the `window_length=0.1` used in `sliding_window()` during training. Using a different window size would give the model a different feature distribution than it was trained on.

---

## Key Files and Artifacts

| File | Location | Purpose |
|------|----------|---------|
| `W.npy` | `data/participantXX/SX/` | Raw PICARD unmixing matrix |
| `whiteM_calib.npy` | `data/participantXX/SX/` *(needs to be saved)* | Calibration whitening matrix |
| `W_eff.npy` | `data/participantXX/SX/` *(needs to be saved)* | Composed operator = W @ whiteM |
| `electrode_order.npy` | `data/participantXX/SX/` | Maps ICA components to muscles |
| `*_blendshapes_ImprovedEnhancedTransformNet*.joblib` | `data/participantXX/SX/` | Trained PyTorch model |
| `scaler_X_*.joblib` | `results/` | Input feature scaler |
| `scaler_Y_*.joblib` | `results/` | Output blendshape scaler |

---

## What Still Needs to Be Done

1. **Modify `classifying_ica_components.py`** — save `whiteM_calib` and compute/save `W_eff = W @ whiteM_calib` at the end of calibration
2. **Write `control_avatar/send_live_to_CS.py`** — real-time inference script using ring buffer + `W_eff` + PyTorch model
3. **Test time synchronization** — between EDF start time and liveCapture `.asset` start time (already handled in `get_time_delta()` in `prepare_data_for_model.py`)