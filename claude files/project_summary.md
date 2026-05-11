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
9. **`control_avatar/send_live_to_CS.py`** — connects to DAU, loads W_eff + model + scalers, starts inference loop

---

## Architecture of the Real-Time Inference Script

`control_avatar/send_live_to_CS.py` is implemented. It:

1. **Loads pre-trained artifacts** from the calibration session:
   - `W_eff.npy` — composed unmixing matrix (= `W @ whiteM_calib`)
   - `electrode_order.npy` — component-to-muscle mapping
   - `model.joblib` — PyTorch `ImprovedEnhancedTransformNet`
   - `scaler_X.joblib`, `scaler_Y.joblib`

2. **Receives EMG** using `DataHandler`, accumulates into a ring buffer (2 seconds, 16 channels) on a background thread

3. **Every 50ms (20 FPS)**, runs inference:
   ```
   filter_signal(last 1s of buffer)     ← notch + bandpass, same as calibration
   W_eff @ filtered_chunk               ← apply fixed unmixing operator
   reorder by electrode_order
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
**Answer:** Yes, but not by applying `W` alone. The PICARD ICA was run on data that was already whitened with a whitening matrix `whiteM_calib` computed from the calibration recording. So `W` was designed to unmix whitened data. The correct operator for raw data is:
```
W_eff = W @ whiteM_calib
```
This `W_eff` is the actual inverse of the physical mixing matrix A (determined by electrode geometry, which is fixed). It should be computed and saved during calibration, and applied directly to filtered real-time chunks.

### Q4: Should you re-whiten each real-time chunk?
**Answer:** No — this is wrong. If you compute a new whitening matrix per chunk, you project the chunk into a different coordinate system than W was calibrated for. Applying W to a differently-whitened chunk gives components that no longer correspond to the same muscles. You would need to re-run PICARD (minutes of computation) to get a valid W for each new whitening. The whole point of calibration is that `W_eff` is a fixed linear operator (the physical inverse mixing matrix). Apply it directly.

### Q5: What about wavelet denoising in real-time?
**Answer:** Wavelet denoising cannot be done in real-time without introducing unacceptable latency. The reason is not speed — it is boundary artifacts. The calibration code processes 10-second windows (e.g., 8000 samples at 800Hz). The db15 wavelet at level 5 has a boundary influence of ~960 samples inward from each edge; with 8000 samples only ~12% of the window is contaminated. In real-time you have at most 1 second of context (500 samples at 500Hz), so ~38% of the window is corrupted. Running denoising on a short window makes the signal worse, not better. To use denoising without artifacts you would need to hold back ~5 seconds of data until it is in the "safe" center of a longer window, making the system unacceptably laggy.

**The correct fix** is to match the calibration preprocessing to what real-time can actually reproduce:
- Remove wavelet denoising from `classifying_ica_components.py:perform_ica_algorithm` (the bandpass filter 35–249Hz already handles the main artifacts it was targeting)
- Also remove the downsample-to-800Hz step (`down_sample_flag=False`) so W_eff is computed on 500Hz data matching the real-time device rate
- Retrain the ML model after re-running ICA calibration with these changes

This eliminates the two main calibration/inference mismatches: sampling rate and denoising. `send_live_to_CS.py` needs no changes — its pipeline already matches what calibration will produce after this fix.

**Why 250Hz is not a good alternative:** The bandpass filter passes 35–249Hz. At 250Hz the Nyquist is 125Hz, cutting off the top half of the EMG frequency band. It would require changing the filter design and would lose meaningful signal.

### Q6: What window size should real-time RMS use?
**Answer:** 100ms (50 samples at 500 Hz), matching the `window_length=0.1` used in `sliding_window()` during training. Using a different window size would give the model a different feature distribution than it was trained on.

---

## Key Files and Artifacts

| File | Location | Purpose |
|------|----------|---------|
| `W.npy` | `data/participantXX/SX/` | Raw PICARD unmixing matrix |
| `whiteM_calib.npy` | `data/participantXX/SX/` | Calibration whitening matrix |
| `W_eff.npy` | `data/participantXX/SX/` | Composed operator = W @ whiteM — used by real-time inference |
| `electrode_order.npy` | `data/participantXX/SX/` | Maps ICA components to muscles |
| `*_blendshapes_ImprovedEnhancedTransformNet*.joblib` | `data/participantXX/SX/` | Trained PyTorch model |
| `scaler_X_*.joblib` | `results/` | Input feature scaler |
| `scaler_Y_*.joblib` | `results/` | Output blendshape scaler |

---

## What Still Needs to Be Done

1. **Modify `classifying_ica_components.py:perform_ica_algorithm`** — remove the downsample-to-800Hz step and the wavelet denoising step so calibration runs at 500Hz on bandpass-filtered data only (matching what `send_live_to_CS.py` receives at inference time). `whiteM` and `W_eff` are already saved.
2. **Retrain** — re-run calibration (`classifying_ica_components.py`) and training (`EMG_to_Avatar_model.py`) after the above change to produce a model and W_eff that match the real-time signal conditions.
3. **Test time synchronization** — between EDF start time and liveCapture `.asset` start time (already handled in `get_time_delta()` in `prepare_data_for_model.py`)