import os
import subprocess
import sys
import time
from moviepy.editor import VideoFileClip
import pygame
from pygame.locals import QUIT
import tkinter as tk
from tkinter import messagebox
from XtrRT.data import Data


# Videos that should NOT be repeated (played once regardless of n_reps)
NON_EXPRESSION_PATTERNS = ['demonstration', 'info', 'introduction', 'break', 'credits']


def is_expression_video(video_file):
    """Return True if the video is an actual expression (not demonstration, intro, break, or credits)."""
    name = video_file.lower()
    return not any(p in name for p in NON_EXPRESSION_PATTERNS)


def get_params_from_gui():
    """Show a GUI window to collect experiment parameters. Returns a dict with keys:
    mode, n_reps, and (if mode=='record') participant_id, session_number, data_path."""
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
                   command=lambda: toggle_record_fields(True)).pack(side=tk.LEFT, padx=5)
    tk.Radiobutton(mode_frame, text="Visualize only", variable=mode_var, value="visualize",
                   command=lambda: toggle_record_fields(False)).pack(side=tk.LEFT, padx=5)

    # Repetitions (always shown)
    tk.Label(root, text="Repetitions per expression:", anchor="e").grid(
        row=2, column=0, sticky="e", padx=(20, 5), pady=5)
    reps_var = tk.StringVar(value="3")
    tk.Entry(root, textvariable=reps_var, width=40).grid(
        row=2, column=1, sticky="w", padx=(5, 20), pady=5)

    # Record-only fields
    record_labels = [
        ("Participant ID:",  "participant_01", "str"),
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

    submit_row = 3 + len(record_labels) + 1

    def on_submit():
        # Validate repetitions
        try:
            n_reps = int(reps_var.get().strip())
            if n_reps < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "'Repetitions' must be a positive integer.")
            return

        params['mode'] = mode_var.get()
        params['n_reps'] = n_reps

        if params['mode'] == 'record':
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


def play_videos(directory, n_reps, data=None):
    # Get all mp4 files sorted by name (numeric prefix order)
    video_files = sorted(f for f in os.listdir(directory) if f.endswith('.mp4'))

    for video_file in video_files:
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

    n_reps = params['n_reps']
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment videos')
    gui_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'real_time_gui', 'newMain.py'
    )

    if params['mode'] == 'record':
        participant_ID = params['participant_id']
        session_number = params['session_number']
        data_path = params['data_path']

        participant_folder = os.path.join(data_path, participant_ID)
        session_folder = os.path.join(participant_folder, f"S{session_number}")
        os.makedirs(session_folder, exist_ok=True)
        edf_file_path = os.path.join(session_folder, f"{participant_ID}_S{session_number}.edf")

        data = Data("127.0.0.1", 20001, verbose=False, timeout_secs=15, save_as=edf_file_path)
        data.start()
        data.add_annotation("Start recording")

        play_videos(directory, n_reps, data=data)
        free_behavior(data=data)

        data.add_annotation("data_start_time: " + str(data.start_time))
        data.add_annotation("stop_recording")
        data.stop()
        print(data.annotations)

    else:  # visualize only
        gui_proc = subprocess.Popen([sys.executable, gui_script])
        print("[Annotation] Start recording")

        play_videos(directory, n_reps)
        free_behavior()

        print("[Annotation] stop_recording")
        gui_proc.wait()

    print('process_terminated')