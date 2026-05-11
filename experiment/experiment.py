import os
import subprocess
import sys
import time
import queue as _queue
import threading
import numpy as np
from moviepy.editor import VideoFileClip
import pygame
from pygame.locals import QUIT
import tkinter as tk
from tkinter import messagebox
import pyedflib

# Add real_time_gui and xtrodes_connector to sys.path so DataHandler can import its siblings
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_gui_dir = os.path.join(_project_root, 'real_time_gui')
_connector_dir = os.path.join(_gui_dir, 'xtrodes_connector')
for _p in (_project_root, _gui_dir, _connector_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from real_time_gui.xtrodes_connector.DataHandler import DataHandler

XTRODES_APP_ID = "Xtrodes.PC.BluetoothLE.DAU_0xzewdaf21npg!BluetoothLE.App"
NUM_CHANNELS = 16


def launch_xtrodes_app():
    print("[Startup] Launching X-trodes PC App - Dev...")
    subprocess.Popen(
        ['powershell', '-Command', f'Start-Process "shell:AppsFolder\\{XTRODES_APP_ID}"']
    )


def wait_for_stable_stream(host, port, min_packets=3, stability_secs=1.0):
    """Block until the X-trodes stream is flowing stably, then disconnect.
    Phase 1 polls indefinitely so the user can take as long as needed to click 'Start Streaming'.
    Phase 2 briefly connects via DataHandler to confirm data packets are actually arriving."""
    import socket

    print("[Startup] Please click 'Start Streaming' in the X-trodes app now.", flush=True)
    print("[Startup] Waiting for X-trodes app to be ready", end="", flush=True)
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                break
        except (socket.error, OSError):
            print(".", end="", flush=True)
            time.sleep(0.15)

    print("\n[Startup] App ready — checking data stream", end="", flush=True)
    monitor_queue = _queue.Queue()
    handler = DataHandler(host, int(port), monitor_queue)
    threading.Thread(target=handler.start, daemon=True).start()

    received = 0
    stable_since = None
    deadline = time.time() + 30
    try:
        while time.time() < deadline:
            try:
                monitor_queue.get(timeout=0.1)
                received += 1
                if stable_since is None:
                    stable_since = time.time()
                print(".", end="", flush=True)
            except _queue.Empty:
                pass
            if (received >= min_packets
                    and stable_since is not None
                    and (time.time() - stable_since) >= stability_secs):
                print(f"\n[Startup] Stream stable — starting experiment.", flush=True)
                return
    finally:
        handler.stop()

    print("\n[Startup] Warning: stream did not stabilize — continuing anyway.", flush=True)


EDF_FLUSH_SECONDS = 300  # flush to disk every 5 minutes


class EDFRecorder:
    """Connects to the X-trodes BLE app via DataHandler, collects EMG samples,
    and writes them to an EDF+ file using periodic flushing (same approach as newMain.py)."""

    RECORD_TYPES_A2 = {0xa2, 0xa2 | 0x8, 0xa2 | 0x4, 0xa2 | 0x8 | 0x4}
    RECORD_TYPES_A0 = {0xa0, 0xa0 | 0x8}

    def __init__(self, host, port, edf_path):
        self.edf_path = edf_path
        self._packet_queue = _queue.Queue()
        self._handler = DataHandler(host, port, self._packet_queue)
        self._edf_writer = None
        self._buffer = [[] for _ in range(NUM_CHANNELS)]
        self._buffer_samples = 0
        self._total_samples_written = 0
        self._pending_annotations = []  # queued before the EDF writer is initialized
        self._sampling_rate = None
        self._start_time = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._collect_thread = None

    def start(self):
        self._start_time = time.time()
        self._running.set()
        self._collect_thread = threading.Thread(target=self._collect, daemon=True)
        self._collect_thread.start()
        self._handler.start()

    def _init_edf_writer(self):
        fs = self._sampling_rate
        ch_info = [{
            'label': f'Ch{i + 1}',
            'dimension': 'uV',
            'sample_frequency': fs,
            'physical_min': -32768.0,
            'physical_max': 32767.0,
            'digital_min': -32768,
            'digital_max': 32767,
            'prefilter': '',
            'transducer': 'fEMG',
        } for i in range(NUM_CHANNELS)]
        self._edf_writer = pyedflib.EdfWriter(self.edf_path, NUM_CHANNELS,
                                              file_type=pyedflib.FILETYPE_EDFPLUS)
        self._edf_writer.setSignalHeaders(ch_info)
        for label in self._pending_annotations:
            self._edf_writer.writeAnnotation(0.0, -1, label)
            print(f"[Annotation @ 0.00s] {label}")
        self._pending_annotations.clear()
        print(f"[EDFRecorder] Writer initialized: {self.edf_path}")

    def _flush_buffer(self):
        if self._edf_writer is None or self._buffer_samples == 0:
            return
        data = [np.array(ch, dtype=np.float64) for ch in self._buffer]
        self._edf_writer.writeSamples(data)
        self._total_samples_written += len(data[0])
        self._buffer = [[] for _ in range(NUM_CHANNELS)]
        self._buffer_samples = 0

    def _collect(self):
        while self._running.is_set():
            try:
                packet = self._packet_queue.get(timeout=0.1)
            except _queue.Empty:
                continue
            if packet.records is None:
                continue
            for record in packet.records.data_records:
                rt = record.record_type
                if rt in self.RECORD_TYPES_A2 or rt in self.RECORD_TYPES_A0:
                    with self._lock:
                        if self._sampling_rate is None:
                            self._sampling_rate = int(record.sampling_rate / record.down_sample)
                            self._init_edf_writer()
                        samples = (record.data_samples_a2 if rt in self.RECORD_TYPES_A2
                                   else record.data_samples)
                        for i in range(NUM_CHANNELS):
                            self._buffer[i].extend(samples[i])
                        self._buffer_samples += len(samples[0])
                        if self._buffer_samples >= EDF_FLUSH_SECONDS * self._sampling_rate:
                            self._flush_buffer()

    def add_annotation(self, label):
        with self._lock:
            if self._edf_writer is None or self._sampling_rate is None:
                self._pending_annotations.append(label)
                print(f"[Annotation queued] {label}")
                return
            onset = (self._total_samples_written + self._buffer_samples) / self._sampling_rate
            self._edf_writer.writeAnnotation(onset, -1, label)
            if not label.startswith("timing_check"):
                print(f"[Annotation @ {onset:.2f}s] {label}")

    @property
    def start_time(self):
        return self._start_time

    def stop(self):
        self._running.clear()
        self._handler.stop()
        if self._collect_thread:
            self._collect_thread.join(timeout=5)
        with self._lock:
            self._flush_buffer()
            if self._edf_writer is not None:
                self._edf_writer.close()
                print(f"[EDFRecorder] Saved {self._total_samples_written} samples "
                      f"@ {self._sampling_rate} Hz → {self.edf_path}")
            else:
                print("[EDFRecorder] No data collected — EDF not written.")


# Videos that should NOT be repeated (played once regardless of n_reps)
NON_EXPRESSION_PATTERNS = ['demonstration', 'info', 'introduction', 'break', 'credits']


def is_expression_video(video_file):
    """Return True if the video is an actual expression (not demonstration, intro, break, or credits)."""
    name = video_file.lower()
    return not any(p in name for p in NON_EXPRESSION_PATTERNS)


def get_params_from_gui():
    """Show a GUI window to collect experiment parameters. Returns a dict with keys:
    mode, n_reps, and (if mode in ('record','test-record')) participant_id, session_number, data_path."""
    params = {}

    root = tk.Tk()
    root.title("Experiment Setup")
    root.resizable(False, False)

    tk.Label(root, text="Experiment Parameters", font=("Arial", 14, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(15, 10), padx=20)

    # Mode selection
    mode_var = tk.StringVar(value="record")
    mode_frame = tk.Frame(root)
    mode_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10))
    tk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=(0, 10))
    tk.Radiobutton(mode_frame, text="Record (EDF)", variable=mode_var, value="record",
                   command=lambda: _on_mode_change("record")).pack(side=tk.LEFT, padx=5)
    tk.Radiobutton(mode_frame, text="Visualize only", variable=mode_var, value="visualize",
                   command=lambda: _on_mode_change("visualize")).pack(side=tk.LEFT, padx=5)
    tk.Radiobutton(mode_frame, text="Test Record", variable=mode_var, value="test-record",
                   command=lambda: _on_mode_change("test-record")).pack(side=tk.LEFT, padx=5)

    # Repetitions (shown only for record / visualize modes)
    reps_label = tk.Label(root, text="Repetitions per expression:", anchor="e")
    reps_label.grid(row=2, column=0, sticky="e", padx=(20, 5), pady=5)
    reps_var = tk.StringVar(value="3")
    reps_entry = tk.Entry(root, textvariable=reps_var, width=40)
    reps_entry.grid(row=2, column=1, sticky="w", padx=(5, 20), pady=5)

    # Record-only fields (shared by 'record' and 'test-record')
    record_labels = [
        ("Participant ID:",  "participant_05", "str"),
        ("Session number:",  "1",              "int"),
        ("Data path:",       r"C:\Users\Hila\OneDrive\מסמכים\fEMG_to_avatar\data", "str"),
    ]
    record_widgets = []  # list of (label_widget, entry_widget, var, dtype)
    for i, (label, default, dtype) in enumerate(record_labels, start=3):
        lbl = tk.Label(root, text=label, anchor="e")
        lbl.grid(row=i, column=0, sticky="e", padx=(20, 5), pady=5)
        var = tk.StringVar(value=default)
        ent = tk.Entry(root, textvariable=var, width=40)
        ent.grid(row=i, column=1, sticky="w", padx=(5, 20), pady=5)
        record_widgets.append((lbl, ent, var, dtype))

    def toggle_record_fields(show):
        state = "normal" if show else "disabled"
        for lbl, ent, _, _ in record_widgets:
            lbl.configure(foreground="black" if show else "gray")
            ent.configure(state=state)

    def toggle_reps_field(show):
        state = "normal" if show else "disabled"
        reps_label.configure(foreground="black" if show else "gray")
        reps_entry.configure(state=state)

    def _on_mode_change(mode):
        toggle_record_fields(mode in ("record", "test-record"))
        toggle_reps_field(mode != "test-record")

    # No-demo checkbox
    no_demo_var = tk.BooleanVar(value=False)
    no_demo_row = 3 + len(record_labels)
    tk.Checkbutton(root, text="No Demo (skip intro/break/credits videos)",
                   variable=no_demo_var).grid(
        row=no_demo_row, column=0, columnspan=2, pady=(5, 0))

    submit_row = no_demo_row + 1

    def on_submit():
        params['mode'] = mode_var.get()

        # Validate repetitions only when relevant
        if params['mode'] != 'test-record':
            try:
                n_reps = int(reps_var.get().strip())
                if n_reps < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "'Repetitions' must be a positive integer.")
                return
            params['n_reps'] = n_reps

        if params['mode'] in ('record', 'test-record'):
            values = []
            for _, _, var, dtype in record_widgets:
                val = var.get().strip()
                if not val:
                    messagebox.showerror("Error", "All fields are required.")
                    return
                if dtype == "int":
                    try:
                        val = int(val)
                        if val < 1:
                            raise ValueError
                    except ValueError:
                        messagebox.showerror("Error", f"'{var.get()}' must be a positive integer.")
                        return
                values.append(val)
            params['participant_id'] = values[0]
            params['session_number'] = values[1]
            params['data_path'] = values[2]

        params['no_demo'] = no_demo_var.get()
        root.destroy()

    tk.Button(root, text="Start Experiment", command=on_submit,
              width=20, bg="#4CAF50", fg="white", font=("Arial", 11, "bold")).grid(
        row=submit_row, column=0, columnspan=2, pady=15)

    root.update_idletasks()
    w = root.winfo_reqwidth()
    h = root.winfo_reqheight()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f"+{x}+{y}")

    root.mainloop()

    if not params:
        print("Experiment cancelled.")
        sys.exit(0)

    return params


def play_videos(directory, n_reps, data=None, no_demo=False):
    # Get all mp4 files sorted by name (numeric prefix order)
    video_files = sorted(f for f in os.listdir(directory) if f.endswith('.mp4'))

    for video_file in video_files:
        if no_demo and not is_expression_video(video_file):
            continue

        video_path = os.path.join(directory, video_file)
        annotation = video_file.split('.')[0]

        repeat = n_reps if is_expression_video(video_file) else 1

        for trial in range(1, repeat + 1):
            label = annotation + f"_trial_{trial}"
            if data is not None:
                data.add_annotation(label)
            if repeat > 1:
                print(f"Now playing: {annotation}  (trial {trial}/{repeat})  [{label}]")
            else:
                print(f"Now playing: {annotation}  [{label}]")

            clip = VideoFileClip(video_path)
            clip.preview()


def free_behavior(data=None):
    if data is not None:
        data.add_annotation("free_behavior")
    else:
        print("[Annotation] free_behavior")
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Experiment Display")
    font = pygame.font.Font(None, 64)
    text = font.render("free behavior", True, (255, 255, 255))
    text_rect = text.get_rect(center=(400, 300))
    start_time = time.time()
    while time.time() - start_time < 120:  # Run for 2 minutes (120 seconds)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        screen.fill((0, 0, 0))
        screen.blit(text, text_rect)
        pygame.display.flip()
    pygame.quit()
    if data is not None:
        data.add_annotation("finished_free_behavior")
    else:
        print("[Annotation] finished_free_behavior")


if __name__ == '__main__':

    params = get_params_from_gui()

    n_reps = params.get('n_reps', 1)
    no_demo = params.get('no_demo', False)
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment videos')
    gui_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'real_time_gui', 'newMain.py'
    )

    launch_xtrodes_app()

    # Run loopback exemption in background — doesn't need to complete before streaming starts
    bat_path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'real_time_gui', 'checknetisolation.bat'
    ))
    threading.Thread(target=lambda: subprocess.run([bat_path], shell=True), daemon=True).start()

    wait_for_stable_stream("127.0.0.1", 20001)

    if params['mode'] == 'record':
        participant_ID = params['participant_id']
        session_number = params['session_number']
        data_path = params['data_path']

        participant_folder = os.path.join(data_path, participant_ID)
        session_folder = os.path.join(participant_folder, f"S{session_number}")
        os.makedirs(session_folder, exist_ok=True)
        edf_file_path = os.path.join(session_folder, f"{participant_ID}_S{session_number}.edf")

        data = EDFRecorder("127.0.0.1", 20001, edf_file_path)
        data.start()
        data.add_annotation("Start recording")

        play_videos(directory, n_reps, data=data, no_demo=no_demo)
        # free_behavior(data=data)

        data.add_annotation("data_start_time: " + str(data.start_time))
        data.add_annotation("stop_recording")
        data.stop()

    elif params['mode'] == 'test-record':
        participant_ID = params['participant_id']
        session_number = params['session_number']
        data_path = params['data_path']

        participant_folder = os.path.join(data_path, participant_ID)
        session_folder = os.path.join(participant_folder, f"S{session_number}")
        os.makedirs(session_folder, exist_ok=True)
        edf_file_path = os.path.join(session_folder,
                                     f"{participant_ID}_S{session_number}_test.edf")

        data = EDFRecorder("127.0.0.1", 20001, edf_file_path)
        data.start()
        data.add_annotation("Start test recording")

        # Send a timestamped annotation every 1 s so latency can be measured offline.
        # Annotation format: "timing_check pc_time=<unix_timestamp_seconds>"
        _stop_timing = threading.Event()

        def _send_timing_annotations():
            while not _stop_timing.wait(1.0):
                data.add_annotation(f"timing_check pc_time={time.time():.6f}")

        timing_thread = threading.Thread(target=_send_timing_annotations, daemon=True)
        timing_thread.start()

        # Show a small window — recording runs until the user clicks Stop
        stop_root = tk.Tk()
        stop_root.title("Test Recording")
        stop_root.resizable(False, False)
        tk.Label(stop_root, text="Recording in progress…",
                 font=("Arial", 13)).pack(padx=30, pady=(20, 10))
        tk.Label(stop_root, text=f"Saving to:\n{edf_file_path}",
                 font=("Arial", 9), fg="gray", wraplength=400, justify="left").pack(
            padx=30, pady=(0, 10))
        tk.Button(stop_root, text="Stop Recording", command=stop_root.destroy,
                  width=20, bg="#e53935", fg="white", font=("Arial", 11, "bold")).pack(
            pady=(0, 20))
        stop_root.update_idletasks()
        w = stop_root.winfo_reqwidth()
        h = stop_root.winfo_reqheight()
        x = (stop_root.winfo_screenwidth() // 2) - (w // 2)
        y = (stop_root.winfo_screenheight() // 2) - (h // 2)
        stop_root.geometry(f"+{x}+{y}")
        stop_root.mainloop()

        _stop_timing.set()
        timing_thread.join(timeout=2)

        data.add_annotation("data_start_time: " + str(data.start_time))
        data.add_annotation("stop_test_recording")
        data.stop()

    else:  # visualize only
        gui_proc = subprocess.Popen([sys.executable, gui_script])
        print("[Annotation] Start recording")

        play_videos(directory, n_reps, no_demo=no_demo)
        # free_behavior()

        print("[Annotation] stop_recording")
        gui_proc.wait()

    print('process_terminated')