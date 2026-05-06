import os
from XtrRT.data import Data
from moviepy.editor import VideoFileClip
from datetime import datetime
import time
import pygame
from pygame.locals import QUIT
import sys
import tkinter as tk
from tkinter import messagebox


# Videos that should NOT be repeated (played once regardless of n_reps)
NON_EXPRESSION_PATTERNS = ['demonstration', 'info', 'introduction', 'break', 'credits']


def is_expression_video(video_file):
    """Return True if the video is an actual expression (not demonstration, intro, break, or credits)."""
    name = video_file.lower()
    return not any(p in name for p in NON_EXPRESSION_PATTERNS)


def get_params_from_gui():
    """Show a GUI window to collect experiment parameters. Returns a dict with keys:
    participant_id, session_number, n_reps, data_path."""
    params = {}

    root = tk.Tk()
    root.title("Experiment Setup")
    root.resizable(False, False)

    tk.Label(root, text="Experiment Parameters", font=("Arial", 14, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(15, 10), padx=20)

    labels_defaults = [
        ("Participant ID:",          "participant_01",                    "str"),
        ("Session number:",          "1",                                 "int"),
        ("Repetitions per expression:", "3",                              "int"),
        ("Data path:",               r"C:\Users\Hila\OneDrive\מסמכים\fEMG_to_avatar\data", "str"),
    ]

    entries = []
    for i, (label, default, _) in enumerate(labels_defaults, start=1):
        tk.Label(root, text=label, anchor="e").grid(row=i, column=0, sticky="e", padx=(20, 5), pady=5)
        var = tk.StringVar(value=default)
        entry = tk.Entry(root, textvariable=var, width=40)
        entry.grid(row=i, column=1, sticky="w", padx=(5, 20), pady=5)
        entries.append((var, labels_defaults[i - 1][2]))

    def on_submit():
        values = []
        for var, dtype in entries:
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
        params['n_reps'] = values[2]
        params['data_path'] = values[3]
        root.destroy()

    tk.Button(root, text="Start Experiment", command=on_submit,
              width=20, bg="#4CAF50", fg="white", font=("Arial", 11, "bold")).grid(
        row=len(labels_defaults) + 1, column=0, columnspan=2, pady=15)

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


def play_videos(directory, data, n_reps):
    # Get all mp4 files sorted by name (numeric prefix order)
    video_files = sorted(f for f in os.listdir(directory) if f.endswith('.mp4'))

    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        annotation = video_file.split('.')[0]

        repeat = n_reps if is_expression_video(video_file) else 1

        for trial in range(1, repeat + 1):
            data.add_annotation(annotation + f"_trial_{trial}")
            if repeat > 1:
                print(f"Now playing: {annotation}  (trial {trial}/{repeat})")
            else:
                print(f"Now playing: {annotation}")

            clip = VideoFileClip(video_path)
            clip.preview()


def free_behavior(data):
    data.add_annotation("free_behavior")
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
    data.add_annotation("finished_free_behavior")


if __name__ == '__main__':

    params = get_params_from_gui()

    participant_ID = params['participant_id']
    session_number = params['session_number']
    n_reps = params['n_reps']
    data_path = params['data_path']

    participant_folder = os.path.join(data_path, participant_ID)
    session_folder = os.path.join(participant_folder, f"S{session_number}")
    if not os.path.exists(participant_folder):
        os.makedirs(participant_folder)
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)
    edf_file_path = os.path.join(session_folder, f"{participant_ID}_S{session_number}.edf")

    host_name = "127.0.0.1"
    port = 20001
    data = Data(host_name, port, verbose=False, timeout_secs=15, save_as=edf_file_path)
    data.start()

    data.add_annotation("Start recording")
    directory = 'experiment videos'
    play_videos(directory, data, n_reps)

    free_behavior(data)

    data.add_annotation("data_start_time: " + str(data.start_time))
    data.add_annotation("stop_recording")
    data.stop()

    print(data.annotations)
    print('process_terminated')