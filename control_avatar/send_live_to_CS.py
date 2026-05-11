"""
Real-time EMG → blendshape inference → Unity TCP sender.

Loads pre-trained ICA and ML artifacts from a calibration session, receives live
EMG from the DAU via DataHandler, and streams blend shapes to Unity at 20 Hz.

Required files in data/<PARTICIPANT_ID>/<SESSION_NUMBER>/:
    <PARTICIPANT_ID>_<SESSION_NUMBER>_<WAVELET>_W_eff.npy
    <PARTICIPANT_ID>_<SESSION_NUMBER>_<WAVELET>_electrode_order.npy
    <PARTICIPANT_ID>_<SESSION_NUMBER>_blendshapes_<MODEL_NAME>_ICA.joblib

Required files in results/:
    scaler_X_<PARTICIPANT_ID>_<SESSION_NUMBER>.joblib
    scaler_Y_<PARTICIPANT_ID>_<SESSION_NUMBER>.joblib
"""

import os
import sys
import socket
import time
import queue as _queue
import threading
import numpy as np
import pandas as pd
import joblib
import torch

# ── path setup ─────────────────────────────────────────────────────────────────
_dir        = os.path.dirname(os.path.abspath(__file__))   # control_avatar/
_project    = os.path.dirname(_dir)                         # project root
_data_proc  = os.path.join(_project, 'data_process')
_gui        = os.path.join(_project, 'real_time_gui')
_connector  = os.path.join(_gui, 'xtrodes_connector')

for _p in (_project, _data_proc, _gui, _connector):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from real_time_gui.xtrodes_connector.DataHandler import DataHandler
from classifying_ica_components import filter_signal
from prepare_data_for_model import normalize_ica_data
from send_data_to_CS import fill_symetrical
from CONSTS import mapping, blend_shapes, relevant_blendshapes

# ── session config — edit these before each session ───────────────────────────
PARTICIPANT_ID = 'participant_03'
SESSION_NUMBER = 'S1'
MODEL_NAME     = 'ImprovedEnhancedTransformNet_trial_1'
WAVELET        = 'db15'

HOST_DAU       = '127.0.0.1'
PORT_DAU       = 20001
PORT_UNITY     = 65432

FS             = 500    # EMG sampling rate (Hz) — must match the DAU setting
SEND_HZ        = 20     # inference + send rate (Hz)
BUFFER_SECS    = 2      # total ring buffer duration
FILTER_SECS    = 1      # context fed to filtfilt to avoid edge artifacts
RMS_SECS       = 0.1    # RMS window — must match window_length used in training
NUM_CHANNELS   = 16

BUFFER_SIZE    = int(FS * BUFFER_SECS)
FILTER_CONTEXT = int(FS * FILTER_SECS)
RMS_WINDOW     = int(FS * RMS_SECS)

RECORD_TYPES_A2 = {0xa2, 0xa2 | 0x8, 0xa2 | 0x4, 0xa2 | 0x8 | 0x4}
RECORD_TYPES_A0 = {0xa0, 0xa0 | 0x8}
# ───────────────────────────────────────────────────────────────────────────────


def load_artifacts(data_path, results_path):
    session_path = os.path.join(data_path, PARTICIPANT_ID, SESSION_NUMBER)

    W_eff = np.load(os.path.join(session_path,
        f'{PARTICIPANT_ID}_{SESSION_NUMBER}_{WAVELET}_W_eff.npy'))

    electrode_order = np.load(os.path.join(session_path,
        f'{PARTICIPANT_ID}_{SESSION_NUMBER}_{WAVELET}_electrode_order.npy')).astype(int)

    model = joblib.load(os.path.join(session_path,
        f'{PARTICIPANT_ID}_{SESSION_NUMBER}_blendshapes_{MODEL_NAME}_ICA.joblib'))
    model = model.cpu()
    model.eval()

    scaler_X = joblib.load(os.path.join(results_path,
        f'scaler_X_{PARTICIPANT_ID}_{SESSION_NUMBER}.joblib'))
    scaler_Y = joblib.load(os.path.join(results_path,
        f'scaler_Y_{PARTICIPANT_ID}_{SESSION_NUMBER}.joblib'))

    print(f'[Artifacts] W_eff {W_eff.shape}, electrode_order, model, scalers loaded.')
    return W_eff, electrode_order, model, scaler_X, scaler_Y


def infer(ring_buf, W_eff, electrode_order, model, scaler_X, scaler_Y):
    """Run one inference step on a snapshot of the ring buffer.

    Pipeline (must match the calibration preprocessing exactly):
      filter → apply W_eff (= W @ whiteM_calib) → reorder → normalize → RMS → scale → model
    """
    # use the last FILTER_CONTEXT samples so filtfilt has enough context
    context = ring_buf[:, -FILTER_CONTEXT:].copy()      # (16, FILTER_CONTEXT)
    filtered = filter_signal(context, FS)                # notch + bandpass, same as calibration

    # apply the composed unmixing operator — no per-chunk whitening needed
    ica = W_eff @ filtered                               # (16, FILTER_CONTEXT)

    ica_ordered = np.zeros_like(ica)
    for i, elec in enumerate(electrode_order):
        if elec != 16:
            ica_ordered[elec, :] = ica[i, :]

    ica_ordered = normalize_ica_data(ica_ordered)

    # RMS over the last RMS_WINDOW samples — matches window_length=0.1 used in training
    rms = np.sqrt(np.mean(ica_ordered[:, -RMS_WINDOW:] ** 2, axis=1))  # (16,)

    x = scaler_X.transform(rms.reshape(1, -1))          # (1, 16)
    with torch.no_grad():
        pred = model(torch.FloatTensor(x)).numpy()       # (1, 31)
    pred = scaler_Y.inverse_transform(pred)              # (1, 31)

    df = pd.DataFrame(pred, columns=relevant_blendshapes)
    full = fill_symetrical(df, mapping, blend_shapes)    # (1, 50) numpy array
    return full[0].astype(np.float32)                    # (50,) float32


def main():
    _script_dir  = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_script_dir)
    data_path    = os.path.join(_project_dir, 'data')
    results_path = os.path.join(_project_dir, 'results')

    W_eff, electrode_order, model, scaler_X, scaler_Y = load_artifacts(
        data_path, results_path)

    # ring buffer shared between the collector thread and the inference loop
    ring_buf         = np.zeros((NUM_CHANNELS, BUFFER_SIZE), dtype=np.float64)
    buf_lock         = threading.Lock()
    samples_received = 0
    buf_ready        = threading.Event()

    packet_queue = _queue.Queue(maxsize=200)
    handler = DataHandler(HOST_DAU, PORT_DAU, packet_queue)

    def _collect():
        """Drain DataHandler packets into the ring buffer."""
        nonlocal ring_buf, samples_received
        while True:
            try:
                packet = packet_queue.get(timeout=1.0)
            except _queue.Empty:
                continue
            if packet.records is None:
                continue
            for record in packet.records.data_records:
                rt = record.record_type
                if rt not in RECORD_TYPES_A2 and rt not in RECORD_TYPES_A0:
                    continue
                samples = (record.data_samples_a2 if rt in RECORD_TYPES_A2
                           else record.data_samples)
                chunk = np.array([samples[i] for i in range(NUM_CHANNELS)],
                                 dtype=np.float64)       # (16, n_samples)
                n = chunk.shape[1]
                with buf_lock:
                    ring_buf = np.roll(ring_buf, -n, axis=1)
                    ring_buf[:, -n:] = chunk
                    samples_received += n
                if not buf_ready.is_set() and samples_received >= FILTER_CONTEXT:
                    buf_ready.set()

    threading.Thread(target=_collect, daemon=True).start()
    handler.start()

    # wait for Unity to connect before starting inference
    print(f'[TCP] Waiting for Unity on port {PORT_UNITY}...')
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', PORT_UNITY))
    server.listen(1)
    conn, addr = server.accept()
    print(f'[TCP] Unity connected from {addr}')

    print(f'[Startup] Waiting for {FILTER_SECS}s of EMG data...')
    buf_ready.wait()
    print('[Inference] Starting 20 Hz loop — press Ctrl+C to stop.')

    interval = 1.0 / SEND_HZ
    try:
        while True:
            t0 = time.perf_counter()

            with buf_lock:
                snap = ring_buf.copy()

            blend = infer(snap, W_eff, electrode_order, model, scaler_X, scaler_Y)
            conn.sendall(blend.tobytes())

            elapsed = time.perf_counter() - t0
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)

    except (KeyboardInterrupt, BrokenPipeError, ConnectionResetError):
        print('[Inference] Stopped.')
    finally:
        handler.stop()
        conn.close()
        server.close()


if __name__ == '__main__':
    main()