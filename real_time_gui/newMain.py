import queue
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import tkinter as tk
from tkinter import ttk
import os
import subprocess
from datetime import datetime
import pyedflib
from pylsl import StreamInfo, StreamOutlet, local_clock, cf_string


mpl.use('TkAgg')

from xtrodes_connector.DataHandler import DataHandler

APP_VERSION = "1.1.4"  # Update this as needed

# App User Model ID for the "X-trodes PC App - Dev" UWP app.
XTRODES_APP_ID = "Xtrodes.PC.BluetoothLE.DAU_0xzewdaf21npg!BluetoothLE.App"

packet_queue = queue.Queue()
num_channels = 16  # Adjust based on your specific data structure
window_size = 2500  # Default window size for "normal"
NUMBER_OF_ROWS = 4
NUMBER_OF_COLUMNS = 4
main_window_active=True
paused = False  # Initialize the paused state
global_factor = 1.0  # Initialize with a default value
last_factor_used = 1.0
LOW_PASS_CUTOFF = 20  # Default cutoff frequency
FILTER_ORDER = 5  # Default filter order

EDF_SAVE_DIR = r"C:\Users\Hila\OneDrive\מסמכים\fEMG_to_avatar\Xtrodes EDF files"
EDF_FLUSH_SECONDS = 300  # flush to disk every 5 minutes
edf_writer = None
edf_buffer = [[] for _ in range(16)]
edf_samples_in_buffer = 0
edf_recording = False
edf_start_time = None
edf_start_lsl_time = None
lsl_outlet = None
edf_total_samples_written = 0


fig, axes = plt.subplots(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, figsize=(15, 10))
lines = [ax.plot([], [])[0] for ax in axes.flatten()]
backgrounds = [None] * len(lines)  # Background storage

# Set titles for each graph
for i, ax in enumerate(axes.flatten()):
    ax.set_title(f'Graph #{i}', fontsize=10, pad=2)  # Add padding to avoid overlap with x-axis

original_data_x = [np.array([]) for _ in range(num_channels)]
original_data_y = [np.array([]) for _ in range(num_channels)]
sampling_rate = 500  # default

CONFIG_FILE = 'config.txt'

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            lines = file.readlines()
            if len(lines) >= 3:
                return lines[0].strip(), lines[1].strip(), lines[2].strip()
    return '', '', ''

def save_config(host, port, factor):
    with open(CONFIG_FILE, 'w') as file:
        file.write(f"{host}\n{port}\n{factor}\n")

def print_elapsed_time(start, label):
    elapsed = (time.time() - start) * 1000  # convert to milliseconds
    print(f"{label}: {elapsed:.2f} ms")

def decimate_data(x_data, y_data, max_points):
    factor = max(1, len(x_data) // max_points)
    return x_data[::factor], y_data[::factor], factor
def init_edf_writer(fs):
    global edf_writer, edf_buffer, edf_samples_in_buffer, edf_start_time, edf_start_lsl_time, lsl_outlet, edf_total_samples_written
    os.makedirs(EDF_SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(EDF_SAVE_DIR, f"recording_{timestamp}.edf")
    edf_writer = pyedflib.EdfWriter(filename, num_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
    edf_start_lsl_time = local_clock()
    edf_start_time = time.time()
    info = StreamInfo('XtrodesMarkers', 'Markers', 1, 0, cf_string, 'xtrodes_markers_001')
    lsl_outlet = StreamOutlet(info)
    print("[LSL] Marker stream outlet created: XtrodesMarkers")
    ch_headers = [{
        'label': f'Ch{i}',
        'dimension': 'uV',
        'sample_frequency': int(fs),
        'physical_max': 32767.0,
        'physical_min': -32768.0,
        'digital_max': 32767,
        'digital_min': -32768,
        'prefilter': '',
        'transducer': ''
    } for i in range(num_channels)]
    edf_writer.setSignalHeaders(ch_headers)
    edf_buffer = [[] for _ in range(num_channels)]
    edf_samples_in_buffer = 0
    edf_total_samples_written = 0
    print(f"[EDF] Recording started: {filename}")

def flush_edf_buffer():
    global edf_buffer, edf_samples_in_buffer, edf_total_samples_written
    if edf_writer is None or edf_samples_in_buffer == 0:
        return
    data = [np.array(ch, dtype=np.float64) for ch in edf_buffer]
    edf_writer.writeSamples(data)
    edf_total_samples_written += len(data[0])
    edf_buffer = [[] for _ in range(num_channels)]
    edf_samples_in_buffer = 0
    print(f"[EDF] Flushed to disk")

def close_edf_writer():
    global edf_writer, lsl_outlet, edf_start_lsl_time
    if edf_writer is not None:
        flush_edf_buffer()
        edf_writer.close()
        edf_writer = None
        print("[EDF] File closed and saved.")
    if lsl_outlet is not None:
        del lsl_outlet
        lsl_outlet = None
        edf_start_lsl_time = None
        print("[LSL] Marker stream closed.")

def add_edf_annotation(description):
    if edf_writer is None:
        return
    lsl_now = local_clock()
    if lsl_outlet is not None:
        lsl_outlet.push_sample([description], lsl_now)
    # Compute onset from sample count — immune to packet loss and clock drift
    onset = (edf_total_samples_written + len(edf_buffer[0])) / sampling_rate
    edf_writer.writeAnnotation(onset, -1, description)
    print(f"[EDF] Annotation at {onset:.3f}s: {description}")

def process_data():
    """Function to process incoming data and update arrays without rendering on main window."""
    global sampling_rate, global_factor, edf_recording, edf_writer, edf_buffer, edf_samples_in_buffer

    packets_to_process = []
    while packet_queue.qsize() > 1:
        packets_to_process.append(packet_queue.get())

    if packets_to_process:
        new_values_per_channel = [[] for _ in range(num_channels)]
        for stream_packet in packets_to_process:
            for record in stream_packet.records.data_records:
                if record.record_type in {0xa2, 0xa0, (0xa2 | 0x8), (0xa0 | 0x8),(0xa2 | 0x4),(0xa2 | 0x8 | 0x4)}:
                    if sampling_rate!=record.sampling_rate / record.down_sample:
                        sampling_rate=record.sampling_rate / record.down_sample
                        set_window_size(window_size)
                    sampling_rate = record.sampling_rate / record.down_sample
                    for i in range(num_channels):
                        if record.record_type in (0xa2,(0xa2 | 0x8)):
                            new_values_per_channel[i].extend(record.data_samples_a2[i])
                        elif record.record_type in (0xa0,(0xa0 | 0x8)):
                            new_values_per_channel[i].extend(record.data_samples[i])

        if global_factor is not None:
            new_values_per_channel = [[y * global_factor for y in channel] for channel in new_values_per_channel]

        # EDF: buffer the same scaled values shown on screen and flush periodically
        if edf_recording and new_values_per_channel[0]:
            if edf_writer is None:
                init_edf_writer(sampling_rate)
            if edf_writer is not None:
                for i in range(num_channels):
                    edf_buffer[i].extend(new_values_per_channel[i])
                edf_samples_in_buffer += len(new_values_per_channel[0])
                if edf_samples_in_buffer >= EDF_FLUSH_SECONDS * sampling_rate:
                    flush_edf_buffer()

        for i in range(num_channels):
            if new_values_per_channel[i]:
                new_x_data = np.arange(len(original_data_x[i]), len(original_data_x[i]) + len(new_values_per_channel[i])) / sampling_rate
                original_data_x[i] = np.concatenate((original_data_x[i], new_x_data))
                original_data_y[i] = np.concatenate((original_data_y[i], new_values_per_channel[i]))

                if len(original_data_x[i]) > window_size:
                    original_data_x[i] = original_data_x[i][-window_size:]
                    original_data_y[i] = original_data_y[i][-window_size:]



def update(frame):
    global main_window_active, zoom_window_active,last_factor_used

    # Process incoming data without rendering to main window if zoom is active

    if main_window_active:
        process_data()

        for i, line in enumerate(lines):
            max_points = int(line.axes.get_window_extent().width)
            decimated_x, decimated_y, decimation_factor = decimate_data(original_data_x[i], original_data_y[i],
                                                                        max_points)
            display_x_data = np.arange(1, len(decimated_x) + 1) * decimation_factor / sampling_rate
            line.set_data(display_x_data, decimated_y)
            axes[i // NUMBER_OF_COLUMNS, i % NUMBER_OF_COLUMNS].set_xlim(0, int(window_size / sampling_rate))
            if backgrounds[i] is None:
                axes[i // NUMBER_OF_COLUMNS, i % NUMBER_OF_COLUMNS].set_ylim(-1000, 10000)
                fig.canvas.draw()
                backgrounds[i] = fig.canvas.copy_from_bbox(axes[i // NUMBER_OF_COLUMNS, i % NUMBER_OF_COLUMNS].bbox)
            if global_factor != last_factor_used:
                autoscale_y(None,axes[i // NUMBER_OF_COLUMNS, i % NUMBER_OF_COLUMNS],line,i)


                # Trigger redraw for main and zoom canvases based on the new factor

        if global_factor != last_factor_used:
            print(f"Factor changed to {global_factor}, updating data and redrawing canvas.")
        last_factor_used = global_factor  # Update the last factor used

    return lines

def autoscale_y(event, ax, line, index):
    x_data, y_data = line.get_data()
    if x_data.size > 0 and y_data.size > 0:
        if global_factor != last_factor_used:
            y_min, y_max = np.min(global_factor*y_data/last_factor_used), np.max(global_factor*y_data/last_factor_used)
        else:
            y_min, y_max = np.min(y_data), np.max(y_data)
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        ax.set_ylim(y_min, y_max)
        fig.canvas.draw()
        backgrounds[index] = fig.canvas.copy_from_bbox(ax.bbox)

    # Clear the queue
    while not packet_queue.empty():
        packet_queue.get()

    fig.canvas.draw()

def autoscale_all():
    #global global_factor,last_factor_used
    #last_factor_used=global_factor-1 #just to make them different
    for i, ax in enumerate(axes.flatten()):
        autoscale_y(None, ax, lines[i], i)
    #last_factor_used = global_factor

def set_window_size(size):
    global window_size
    final_window_size = int(size * sampling_rate)


    window_size = final_window_size
    print(f"Updated window size: {window_size}")
    global paused
    if not paused:
        # Rescale x-axis and save new background
        for i, ax in enumerate(axes.flatten()):
            ax.set_xlim(0, int(window_size / sampling_rate))
            fig.canvas.draw()
            backgrounds[i] = fig.canvas.copy_from_bbox(ax.bbox)



# Tkinter UI for IP/URL and port input and autoscale buttons
def get_scaling_factor():
    import ctypes

    """Get Windows scaling factor using ctypes."""
    # On Windows 10 and later, we can use DPI awareness APIs to get accurate scaling
    user32 = ctypes.windll.user32
    dc = user32.GetDC(0)
    dpi = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)  # Get DPI (88 is LOGPIXELSX)
    ctypes.windll.user32.ReleaseDC(0, dc)
    return dpi / 96  # 96 is the standard DPI, so this gives us the scaling factor


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        main_window_active = True  # Track whether the main window should update

        self.title(f"Xtrodes RTGraphs Version: {APP_VERSION}")
        self.state('normal')  # Open the window maximized

        self.left_frame = ttk.Frame(self)
        self.left_frame.grid(row=0, column=0, sticky="ns")

        self.left_canvas = tk.Canvas(self.left_frame)
        self.scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.left_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.left_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.left_canvas.configure(
                scrollregion=self.left_canvas.bbox("all")
            )
        )

        self.left_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.left_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.label_ip = ttk.Label(self.scrollable_frame, text="IP/URL:")
        self.label_ip.pack(pady=5, padx=10, anchor="w")
        self.entry_ip = ttk.Entry(self.scrollable_frame)
        self.entry_ip.pack(pady=5, padx=10, fill=tk.X)

        self.label_port = ttk.Label(self.scrollable_frame, text="Port:")
        self.label_port.pack(pady=5, padx=10, anchor="w")
        self.entry_port = ttk.Entry(self.scrollable_frame)
        self.entry_port.pack(pady=5, padx=10, fill=tk.X)

        self.button_connect = ttk.Button(self.scrollable_frame, text="Connect", command=self.connect)
        self.button_connect.pack(pady=(5, 15), padx=10)

        self.label_max_y = ttk.Label(self.scrollable_frame, text="Max Y:")
        self.label_max_y.pack(pady=5, padx=10, anchor="w")
        self.entry_max_y = ttk.Entry(self.scrollable_frame)
        self.entry_max_y.pack(pady=5, padx=10, fill=tk.X)

        self.label_min_y = ttk.Label(self.scrollable_frame, text="Min Y:")
        self.label_min_y.pack(pady=5, padx=10, anchor="w")
        self.entry_min_y = ttk.Entry(self.scrollable_frame)
        self.entry_min_y.pack(pady=5, padx=10, fill=tk.X)

        self.button_set_y = ttk.Button(self.scrollable_frame, text="Set Y Limits", command=self.set_y_limits)
        self.button_set_y.pack(pady=(5, 15), padx=10)

        self.label_factor = ttk.Label(self.scrollable_frame, text="Factor:")
        self.label_factor.pack(pady=5, padx=10, anchor="w")
        self.entry_factor = ttk.Entry(self.scrollable_frame)
        self.entry_factor.pack(pady=5, padx=10, fill=tk.X)

        def set_global_factor():
            global global_factor
            try:
                global_factor = float(self.entry_factor.get())
                print(f"Global factor set to: {global_factor}")
            except ValueError:
                print("Invalid factor input")
        # Button to set the global factor
        self.button_set_factor = ttk.Button(self.scrollable_frame, text="Set Factor", command=set_global_factor)
        self.button_set_factor.pack(pady=5, padx=10)
        #self.entry_factor.bind("<Return>", self.save_factor)  # Save factor on Enter key press

        self.label_sampling_rate = ttk.Label(self.scrollable_frame, text="Sampling Rate:")
        self.label_sampling_rate.pack(pady=5, padx=10, anchor="w")
        self.entry_sampling_rate = ttk.Entry(self.scrollable_frame)
        self.entry_sampling_rate.pack(pady=5, padx=10, fill=tk.X)
        self.entry_sampling_rate.config(state='disabled')  # Disable input to the sampling rate entry

        self.button_autoscale_all = ttk.Button(self.scrollable_frame, text="Autoscale All", command=autoscale_all)
        self.button_autoscale_all.pack(pady=10, padx=10)

        self.button_record = tk.Button(self.scrollable_frame, text="Start Recording",
                                       bg="green", fg="white", font=("TkDefaultFont", 10, "bold"),
                                       command=self.toggle_recording)
        self.button_record.pack(pady=10, padx=10, fill=tk.X)

        self.label_rec_timer = ttk.Label(self.scrollable_frame, text="", foreground="gray")
        self.label_rec_timer.pack(pady=(0, 5), padx=10)
        self._rec_timer_running = False

        ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(self.scrollable_frame, text="Annotations:").pack(pady=(0, 5), padx=10, anchor='w')

        preset_markers = ["Stimulation Start", "Stimulation End", "Blink", "Eye Open", "Eye Close"]
        self.annotation_buttons = []
        for label in preset_markers:
            btn = ttk.Button(self.scrollable_frame, text=label,
                             command=lambda l=label: add_edf_annotation(l),
                             state='disabled')
            btn.pack(pady=2, padx=10, fill=tk.X)
            self.annotation_buttons.append(btn)

        self.entry_annotation = ttk.Entry(self.scrollable_frame)
        self.entry_annotation.pack(pady=(8, 2), padx=10, fill=tk.X)
        self.entry_annotation.insert(0, "Custom note...")
        self.entry_annotation.config(state='disabled')

        self.button_mark = ttk.Button(self.scrollable_frame, text="Add Note",
                                      command=self.add_custom_annotation, state='disabled')
        self.button_mark.pack(pady=(2, 10), padx=10, fill=tk.X)

        ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=(0, 10))

        self.label_window_size = ttk.Label(self.scrollable_frame, text="Signal length (Seconds):")
        self.label_window_size.pack(pady=(15, 5), padx=10, anchor='w')

        self.button_short = ttk.Button(self.scrollable_frame, text="2 Seconds", command=lambda: set_window_size(2))
        self.button_short.pack(pady=5, padx=10, fill=tk.X)
        self.button_normal = ttk.Button(self.scrollable_frame, text="5 Seconds", command=lambda: set_window_size(5))
        self.button_normal.pack(pady=5, padx=10, fill=tk.X)
        self.button_long = ttk.Button(self.scrollable_frame, text="10 Seconds", command=lambda: set_window_size(10))
        self.button_long.pack(pady=(5, 15), padx=10, fill=tk.X)

        self.label_custom_window_size = ttk.Label(self.scrollable_frame, text="Custom Window Size (Seconds):")
        self.label_custom_window_size.pack(pady=5, padx=10, anchor='w')
        self.entry_custom_window_size = ttk.Entry(self.scrollable_frame)
        self.entry_custom_window_size.pack(pady=5, padx=10, fill=tk.X)
        self.entry_custom_window_size.bind("<Return>", self.set_custom_window_size)

        self.host, self.port, self.factor = load_config()
        self.entry_ip.insert(0, self.host)
        self.entry_port.insert(0, self.port)
        self.entry_factor.insert(0, self.factor)

        self.create_autoscale_buttons()

        self.center_frame = ttk.Frame(self)
        self.center_frame.grid(row=0, column=1, sticky="nsew")

        self.canvas = FigureCanvasTkAgg(fig, master=self.center_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle window close event

        self.data_handler = None  # Initialize the data handler attribute

    def create_autoscale_buttons(self):
        self.autoscale_buttons = []
        self.zoom_buttons = []

        for i, ax in enumerate(axes.flatten()):
            button_frame = ttk.Frame(self.scrollable_frame)  # Create a frame for each row of buttons
            button_frame.pack(fill=tk.X, pady=5)  # Use pack to layout the frame

            # Autoscale button
            autoscale_button = ttk.Button(button_frame, text=f'Autoscale {i}',
                                          command=lambda a=ax, l=lines[i], idx=i: autoscale_y(None, a, l, idx))
            autoscale_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)  # Align to the left
            self.autoscale_buttons.append(autoscale_button)

            # Zoom button
            zoom_button = ttk.Button(button_frame, text=f'Zoom {i}',
                                     command=lambda idx=i: self.zoom_channel(idx))
            zoom_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)  # Align to the left
            self.zoom_buttons.append(zoom_button)

    import ctypes

    # Enable DPI awareness (for Windows scaling issues)
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Make the app DPI aware

    
    def zoom_channel(self, channel_index):
        global main_window_active, zoom_window_active,global_factor,LOW_PASS_CUTOFF,FILTER_ORDER
        main_window_active = False
        zoom_window_active = True
        global paused
        paused = False  # Initialize the paused state
        filter_enabled = False  # Variable to track filter state
        # Add filter toggle buttons
        notch_enabled = False
        low_pass_enabled = False
        selected_time_span = 10
        time_span_needed = False
        is_dragging = False
        # Define zoom-specific filter parameters with default values
        zoom_cutoff = LOW_PASS_CUTOFF
        zoom_order = FILTER_ORDER

        zoom_window = tk.Toplevel(self)
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()  # Ensures the program is DPI-aware
        screen_width = user32.GetSystemMetrics(0)  # Scaled screen width
        screen_height = user32.GetSystemMetrics(1)  # Scaled screen height

        #zoom_window.geometry("800x600")
        zoom_window.title(f"Zoomed View - Channel {channel_index}")
        zoom_window.state('zoomed')  # This makes the window open in maximized mode

        sidebar_width = 150  # Fixed width for the sidebar

        # Sidebar frame for buttons with fixed size and scrollbar
        sidebar = ttk.Frame(zoom_window, width=sidebar_width)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)  # Prevent sidebar from resizing



        def toggle_pause():
            global paused
            paused = not paused
            pause_button.config(text="Resume" if paused else "Pause")

        pause_button = ttk.Button(sidebar, text="Pause", command=toggle_pause)
        pause_button.pack(padx=10, pady=10)

        def autoscale_zoom():
            """Autoscale function to dynamically adjust y-limits based on current data."""
            x_data, y_data = zoom_line.get_data()
            if x_data.size > 0 and y_data.size > 0:
                if last_factor_used != global_factor:
                    y_min, y_max = np.min(global_factor*y_data/last_factor_used), np.max(global_factor*y_data/last_factor_used)
                else:
                    y_min, y_max = np.min(y_data), np.max(y_data)
                y_range = y_max - y_min
                y_min -= 0.1 * y_range
                y_max += 0.1 * y_range
                zoom_ax.set_ylim(y_min, y_max)
                line1[0].set_ydata([y_max-0.2*(y_max-y_min), y_max-0.2*(y_max-y_min)])
                line2[0].set_ydata([y_min + 0.2 * (y_max - y_min), y_min + 0.2 * (y_max - y_min)])
                zoom_canvas.draw()

        # Autoscale button
        autoscale_button = tk.Button(sidebar, text="Autoscale", command=lambda: autoscale_zoom())
        autoscale_button.pack(pady=5)

        # Add a button to toggle the filter
        def toggle_low_pass():
            nonlocal low_pass_enabled
            low_pass_enabled = not low_pass_enabled
            low_pass_button.config(text="Disable LP Filter" if low_pass_enabled else "Enable LP Filter")

            # Disable band-pass controls if low-pass is enabled
            if low_pass_enabled:
                band_pass_button.config(state="disabled")
            else:
                band_pass_button.config(state="normal")

        def toggle_notch():
            nonlocal notch_enabled
            notch_enabled = not notch_enabled
            notch_button.config(text="Disable Notch" if notch_enabled else "Enable Notch")






        notch_button = ttk.Button(sidebar, text="Enable Notch", command=toggle_notch)
        notch_button.pack(pady=5)

        def apply_zoom_filter_params():
            nonlocal zoom_cutoff, zoom_order
            try:
                zoom_cutoff = float(entry_zoom_cutoff.get())
                zoom_order = int(entry_zoom_order.get())
                print(f"Zoom Filter - Cutoff: {zoom_cutoff} Hz, Order: {zoom_order}")
                process_data()  # Redraw canvas with updated filter parameters
            except ValueError:
                print("Invalid input for zoom filter parameters")

        # Add Low-pass Filter Cutoff Frequency Input for Zoom Window
        # Low-pass section
        style = ttk.Style()
        style.configure("Thick.TLabelframe", borderwidth=30, relief="solid")
        style.configure("Thick.TLabelframe.Label", font=("Segoe UI", 10, "bold"))

        lowpass_frame = ttk.LabelFrame(sidebar, text="Low-pass Filter", style="Thick.TLabelframe")
        lowpass_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        low_pass_button = ttk.Button(lowpass_frame, text="Enable LP Filter", command=toggle_low_pass)
        low_pass_button.pack(pady=5, padx=5, fill=tk.X)

        label_zoom_cutoff = ttk.Label(lowpass_frame, text="Cutoff (Hz):")
        label_zoom_cutoff.pack(pady=5, padx=5, anchor="w")
        entry_zoom_cutoff = ttk.Entry(lowpass_frame)
        entry_zoom_cutoff.insert(0, str(zoom_cutoff))
        entry_zoom_cutoff.pack(pady=5, padx=5, fill=tk.X)

        label_zoom_order = ttk.Label(lowpass_frame, text="Order:")
        label_zoom_order.pack(pady=5, padx=5, anchor="w")
        entry_zoom_order = ttk.Entry(lowpass_frame)
        entry_zoom_order.insert(0, str(zoom_order))
        entry_zoom_order.pack(pady=5, padx=5, fill=tk.X)

        button_apply_zoom_filter = ttk.Button(lowpass_frame, text="Apply LP Params", command=apply_zoom_filter_params)
        button_apply_zoom_filter.pack(pady=5, padx=5, fill=tk.X)

        # Band-pass filter toggling
        band_pass_enabled = False  # New flag

        # Create Band-pass Section Frame
        bandpass_frame = ttk.LabelFrame(sidebar, text="Band-pass Filter", style="Thick.TLabelframe")
        bandpass_frame.pack(fill=tk.X, padx=10, pady=(20, 5))  # <-- More padding to separate from Low-pass

        label_band_low = ttk.Label(bandpass_frame, text="Low Cutoff (Hz):")
        label_band_low.pack(pady=5, padx=5, anchor="w")
        entry_band_low = ttk.Entry(bandpass_frame)
        entry_band_low.insert(0, "1.0")
        entry_band_low.pack(pady=5, padx=5, fill=tk.X)

        label_band_high = ttk.Label(bandpass_frame, text="High Cutoff (Hz):")
        label_band_high.pack(pady=5, padx=5, anchor="w")
        entry_band_high = ttk.Entry(bandpass_frame)
        entry_band_high.insert(0, "40.0")
        entry_band_high.pack(pady=5, padx=5, fill=tk.X)

        label_band_order = ttk.Label(bandpass_frame, text="Order:")
        label_band_order.pack(pady=5, padx=5, anchor="w")
        entry_band_order = ttk.Entry(bandpass_frame)
        entry_band_order.insert(0, "5")
        entry_band_order.pack(pady=5, padx=5, fill=tk.X)

        def toggle_band_pass():
            nonlocal band_pass_enabled
            band_pass_enabled = not band_pass_enabled
            band_pass_button.config(text="Disable Band-pass" if band_pass_enabled else "Enable Band-pass")

            # Disable low-pass controls if band-pass is enabled
            if band_pass_enabled:
                low_pass_button.config(state="disabled")
            else:
                low_pass_button.config(state="normal")

        band_pass_button = ttk.Button(bandpass_frame, text="Enable Band-pass", command=toggle_band_pass)
        band_pass_button.pack(pady=5, padx=5, fill=tk.X)

        # label_band_low = ttk.Label(sidebar, text="Band-pass Low Cutoff (Hz):")
        # label_band_low.pack(pady=5, padx=10, anchor="w")
        # entry_band_low = ttk.Entry(sidebar)
        # entry_band_low.insert(0, "1.0")
        # entry_band_low.pack(pady=5, padx=10, fill=tk.X)
        #
        # label_band_high = ttk.Label(sidebar, text="Band-pass High Cutoff (Hz):")
        # label_band_high.pack(pady=5, padx=10, anchor="w")
        # entry_band_high = ttk.Entry(sidebar)
        # entry_band_high.insert(0, "40.0")
        # entry_band_high.pack(pady=5, padx=10, fill=tk.X)
        #
        # label_band_order = ttk.Label(sidebar, text="Band-pass Order:")
        # label_band_order.pack(pady=5, padx=10, anchor="w")
        # entry_band_order = ttk.Entry(sidebar)
        # entry_band_order.insert(0, "5")
        # entry_band_order.pack(pady=5, padx=10, fill=tk.X)
        #
        # def toggle_band_pass():
        #     nonlocal band_pass_enabled
        #     band_pass_enabled = not band_pass_enabled
        #     band_pass_button.config(text="Disable Band-pass" if band_pass_enabled else "Enable Band-pass")
        #
        #     # Disable low-pass controls if band-pass is enabled
        #     if band_pass_enabled:
        #         low_pass_button.config(state="disabled")
        #     else:
        #         low_pass_button.config(state="normal")
        #
        # band_pass_button = ttk.Button(sidebar, text="Enable Band-pass", command=toggle_band_pass)
        # band_pass_button.pack(pady=5)

        def set_zoom_time_span(time_span):
            """Set the time span for the zoom graph and mark it for update."""
            nonlocal selected_time_span
            selected_time_span = time_span
            #global window_size
            #window_size=time_span*sampling_rate
            nonlocal time_span_needed
            time_span_needed = True

        # Time span buttons for the zoom window
        self.button_short_span = ttk.Button(sidebar, text="2 Seconds",
                                            command=lambda: set_zoom_time_span(2))
        self.button_short_span.pack(pady=5, padx=10, fill=tk.X)

        self.button_normal_span = ttk.Button(sidebar, text="5 Seconds",
                                             command=lambda: set_zoom_time_span(5))
        self.button_normal_span.pack(pady=5, padx=10, fill=tk.X)

        self.button_long_span = ttk.Button(sidebar, text="10 Seconds",
                                           command=lambda: set_zoom_time_span(10))
        self.button_long_span.pack(pady=(5, 15), padx=10, fill=tk.X)



        # Get screen dimensions for dynamic sizing
        scaling_factor = get_scaling_factor()
        screen_width = int(zoom_window.winfo_screenwidth() / scaling_factor)
        screen_height = int(zoom_window.winfo_screenheight() / scaling_factor)

        dpi = zoom_window.winfo_fpixels('1i')
        margin = 0

        fig_width, fig_height = int((screen_width-margin) / dpi), int((screen_height-margin) / dpi)  # Adjust size in inches based on screen size

        zoom_fig = plt.Figure(figsize=(fig_width, fig_height), dpi=dpi)
        zoom_ax = zoom_fig.add_subplot(111)

        # Enable the grid on the zoomed graph
        zoom_ax.grid(True)  # Basic grid
        # Optional: Customize grid appearance
        zoom_ax.grid(color='gray', linestyle='--', linewidth=0.5)

        # Create the line object specifically for the zoom window
        zoom_line, = zoom_ax.plot([], [], animated=True,linewidth=0.5)  # Set animated=True for blit

        zoom_canvas = FigureCanvasTkAgg(zoom_fig, master=zoom_window)
        zoom_canvas.draw()
        zoom_canvas.get_tk_widget().grid(row=0, column=1, sticky="new")

        # Configure layout grid to adjust figure size dynamically
        zoom_window.grid_columnconfigure(1, weight=1)  # Column 1 expands
        zoom_window.grid_rowconfigure(0, weight=0)  # Row 0 expands

        # Get y-limits from the corresponding main window plot and set them
        main_ax = axes[channel_index // NUMBER_OF_COLUMNS, channel_index % NUMBER_OF_COLUMNS]
        y_min, y_max = main_ax.get_ylim()
        zoom_ax.set_ylim(y_min, y_max)

        line1 = None
        line2 = None
        dragging_line = None

        line1 = zoom_ax.plot([0, window_size / sampling_rate], [y_max-0.2*(y_max-y_min), y_max-0.2*(y_max-y_min)], color="red", linewidth=0.5)
        line2 = zoom_ax.plot([0, window_size / sampling_rate], [y_min+0.2*(y_max-y_min), y_min+0.2*(y_max-y_min)], color="blue", linewidth=0.5)
        def on_click(event):
            nonlocal dragging_line,is_dragging,line1,line2
            #if paused:
            """Identify the closest line to the mouse click for dragging."""
            if event.inaxes != line1[0].axes:
                return
            # Check distances to line1 and line2, selecting the closest line
            if abs(event.ydata - line1[0].get_ydata()[0]) < abs(event.ydata - line2[0].get_ydata()[0]):
                dragging_line = line1
            else:
                dragging_line = line2
            is_dragging = True  # Set dragging mode to true
            return line1, line2

        def on_drag(event):
            nonlocal dragging_line
            #if paused:
            """Drag the line vertically if one is selected."""
            if is_dragging and dragging_line and event.inaxes == dragging_line[0].axes:
                #ani.event_source.stop()
                # Update the y-data of the dragged line to follow the mouse
                y_pos = event.ydata
                dragging_line[0].set_ydata([y_pos, y_pos])

                update_distance_text()
                #event.canvas.figure.canvas.draw()
                #zoom_fig.canvas.draw_idle()
                #zoom_fig.canvas.flush_events()

            return dragging_line

        def on_release(event):
            nonlocal dragging_line,is_dragging
            """Stop dragging when the mouse is released."""
            dragging_line = None
            is_dragging = False
            #ani.event_source.start()

        def update_distance_text():
            """Calculate and display the vertical distance between the two lines in real y-values."""
            y1 = line1[0].get_ydata()[0]
            y2 = line2[0].get_ydata()[0]
            y_distance = abs(y1 - y2)
            distance_label.config(text=f"Y Difference: {y_distance:.2f}")







        # Display distance text
        distance_label = ttk.Label(sidebar, text="Y Difference: 0.00", width=20, anchor="w")
        distance_label.pack(pady=5, padx=10, anchor="w")

        # Adding Max Y and Min Y entry fields in the sidebar
        def set_zoom_y_limits():
            try:
                min_y = float(entry_zoom_min_y.get())
                max_y = float(entry_zoom_max_y.get())
                zoom_ax.set_ylim(min_y, max_y)
                zoom_canvas.draw()
            except ValueError:
                print("Invalid input for Y limits")

        # Max Y input
        label_zoom_max_y = ttk.Label(sidebar, text="Max Y:")
        label_zoom_max_y.pack(pady=5, padx=10, anchor="w")
        entry_zoom_max_y = ttk.Entry(sidebar)
        entry_zoom_max_y.pack(pady=5, padx=10, fill=tk.X)

        # Min Y input
        label_zoom_min_y = ttk.Label(sidebar, text="Min Y:")
        label_zoom_min_y.pack(pady=5, padx=10, anchor="w")
        entry_zoom_min_y = ttk.Entry(sidebar)
        entry_zoom_min_y.pack(pady=5, padx=10, fill=tk.X)

        # Button to apply Y limits to zoomed graph
        button_set_zoom_y = ttk.Button(sidebar, text="Set Y Limits", command=set_zoom_y_limits)
        button_set_zoom_y.pack(pady=(5, 15), padx=10)

        # Factor input in the zoom sidebar
        label_zoom_factor = ttk.Label(sidebar, text="Factor:")
        label_zoom_factor.pack(pady=5, padx=10, anchor="w")

        entry_zoom_factor = ttk.Entry(sidebar)
        entry_zoom_factor.pack(pady=5, padx=10, fill=tk.X)
        entry_zoom_factor.insert(0, str(global_factor))  # Display current global factor

        def set_zoom_factor():
            global global_factor,last_factor_used
            try:
                global_factor = float(entry_zoom_factor.get())
                autoscale_zoom()
                last_factor_used = global_factor
            except ValueError:
                print("Invalid factor input for zoom")

        # Button to set the global factor from zoom window
        button_set_zoom_factor = ttk.Button(sidebar, text="Set Factor", command=set_zoom_factor)
        button_set_zoom_factor.pack(pady=5, padx=10)

        zoom_canvas.mpl_connect("button_press_event", on_click)
        zoom_canvas.mpl_connect("motion_notify_event", on_drag)
        zoom_canvas.mpl_connect("button_release_event", on_release)


        def init_zoom():
            """Initialize zoom line for blitting."""
            global window_size,sampling_rate
            nonlocal selected_time_span
            zoom_line.set_data([], [])
            selected_time_span=int(window_size/sampling_rate)

            return zoom_line,line1[0],line2[0]

        from scipy.signal import butter, filtfilt, iirnotch

        def butter_lowpass(cutoff, fs, order=5):
            nyquist = 0.5 * fs  # Nyquist frequency
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def lowpass_filter(data, cutoff=20, sampling_rate=500, order=5):
            b, a = butter_lowpass(cutoff, sampling_rate, order=order)
            y = filtfilt(b, a, data)

            return y

        def bandpass_filter(data, lowcut, highcut, fs, order=5):
            from scipy.signal import butter, filtfilt
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, data)

        # Notch filter design (50 Hz notch filter)
        def notch_filter(data, freq=50, fs=500, q=30):
            b, a = iirnotch(freq / (fs / 2), q)
            return filtfilt(b, a, data)

        def update_zoom(frame):
            process_data()  # Continue processing data even when paused

            global paused,last_factor_used
            if not paused:
                # Only update the line if not paused
                # Apply the low-pass filter only if filter_enabled is True
                data = original_data_y[channel_index]
                if notch_enabled:
                    data = notch_filter(data, freq=50, fs=sampling_rate)

                if low_pass_enabled:
                    data = lowpass_filter(data, cutoff=zoom_cutoff, sampling_rate=sampling_rate,order=zoom_order)
                if band_pass_enabled:
                    try:
                        lowcut = float(entry_band_low.get())
                        highcut = float(entry_band_high.get())
                        order = int(entry_band_order.get())
                        data = bandpass_filter(data, lowcut, highcut, sampling_rate, order=order)
                    except Exception as e:
                        print(f"[Bandpass Error] {e}")

                # y_data = lowpass_filter(original_data_y[channel_index],sampling_rate=sampling_rate) if filter_enabled else original_data_y[
                #    channel_index]

                max_points = int(zoom_ax.get_window_extent().width)
                decimated_x, decimated_y, decimation_factor = decimate_data(
                    original_data_x[channel_index], data, max_points
                )
                display_x_data = np.arange(1, len(decimated_x) + 1) * decimation_factor / sampling_rate
                zoom_line.set_data(display_x_data, decimated_y)
                #zoom_ax.set_xlim(0, window_size / sampling_rate)
                nonlocal  time_span_needed
                nonlocal  selected_time_span
                if time_span_needed:
                    zoom_ax.set_xlim(0, int(selected_time_span*sampling_rate / sampling_rate))
                    zoom_fig.canvas.draw()
                    time_span_needed=False
                    set_window_size(selected_time_span)
                else:
                    zoom_ax.set_xlim(0, int(selected_time_span*sampling_rate / sampling_rate))
                    #zoom_ax.set_xlim(0, window_size / sampling_rate)
                # if global_factor != last_factor_used:
                #     last_factor_used = global_factor  # Update the last factor used
                #     print(f"Factor changed to {global_factor}, updating data and redrawing canvas.")
                #
                #     # Trigger redraw for main and zoom canvases based on the new factor
                #
                #     zoom_canvas.draw()  # Redraw the zoom window canvas


            return zoom_line,line1[0],line2[0]

        ani = FuncAnimation(
            zoom_fig, update_zoom, init_func=init_zoom, interval=50, blit=True
        )
        zoom_window.ani = ani

        # Make sure close_zoom is called on closing
        zoom_window.protocol("WM_DELETE_WINDOW", lambda: self.close_zoom(zoom_window))

    def close_zoom(self, zoom_window):
        global main_window_active, zoom_window_active
        main_window_active = True
        zoom_window_active = False
        if hasattr(zoom_window, 'ani'):
            zoom_window.ani.event_source.stop()
        zoom_window.destroy()

    def on_closing(self):
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
        self.quit()
        self.destroy()




    def connect(self):
        self.host = self.entry_ip.get() #localhost
        self.port = int(self.entry_port.get())#20001
        save_config(self.host, self.port, self.entry_factor.get())

        if self.host and self.port:
            self.button_connect.config(state='disabled')  # prevent multiple connections
            data_handler_thread = threading.Thread(target=start_data_handler, args=(self.host, self.port))
            data_handler_thread.daemon = True
            data_handler_thread.start()

            # Start the animation within Tkinter's main loop
            ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
            self.ani = ani  # Store the animation reference in the app

    def set_y_limits(self):
        try:
            min_y = float(self.entry_min_y.get())
            max_y = float(self.entry_max_y.get())
            for ax in axes.flatten():
                ax.set_ylim(min_y, max_y)
            fig.canvas.draw()
        except ValueError:
            print("Invalid input for y-limits")

    def get_factor(self):
        try:
            return float(self.entry_factor.get())
        except ValueError:
            print("Invalid input for factor")
            return None

    def save_factor(self, event=None):
        factor = self.entry_factor.get()
        save_config(self.host, self.port, factor)
        print(f"Saved factor: {factor}")
        autoscale_all()  # Perform a full autoscale after saving the factor

    def get_sampling_rate(self):
        try:
            return float(self.entry_sampling_rate.get())
        except ValueError:
            print("Invalid input for sampling rate")
            return None

    def update_sampling_rate(self, rate):
        self.entry_sampling_rate.config(state='normal')
        self.entry_sampling_rate.delete(0, tk.END)
        self.entry_sampling_rate.insert(0, str(rate))
        self.entry_sampling_rate.config(state='disabled')

    def set_custom_window_size(self, event=None):
        try:
            custom_size = float(self.entry_custom_window_size.get())
            if custom_size > 0:
                final_window_size = int(custom_size)
                set_window_size(final_window_size)
        except ValueError:
            print("Invalid input for custom window size")

    def toggle_recording(self):
        global edf_recording
        if not edf_recording:
            edf_recording = True
            self.button_record.config(text="Stop Recording", bg="red")
            for btn in self.annotation_buttons:
                btn.config(state='normal')
            self.entry_annotation.config(state='normal')
            self.entry_annotation.delete(0, tk.END)
            self.button_mark.config(state='normal')
            self._rec_timer_running = True
            self._rec_start = time.time()
            self._update_recording_timer()
            print("[EDF] Recording started by user.")
        else:
            edf_recording = False
            self._rec_timer_running = False
            elapsed = time.time() - self._rec_start
            close_edf_writer()
            self.button_record.config(text="Start Recording", bg="green")
            for btn in self.annotation_buttons:
                btn.config(state='disabled')
            self.entry_annotation.config(state='disabled')
            self.entry_annotation.delete(0, tk.END)
            self.entry_annotation.insert(0, "Custom note...")
            self.button_mark.config(state='disabled')
            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            self.label_rec_timer.config(
                text=f"Last recording: {h:02d}:{m:02d}:{s:02d}", foreground="gray"
            )
            print("[EDF] Recording stopped by user.")

    def _update_recording_timer(self):
        if not self._rec_timer_running:
            return
        elapsed = time.time() - self._rec_start
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        self.label_rec_timer.config(
            text=f"REC  {h:02d}:{m:02d}:{s:02d}", foreground="red"
        )
        self.after(1000, self._update_recording_timer)

    def add_custom_annotation(self):
        text = self.entry_annotation.get().strip()
        if text:
            add_edf_annotation(text)

    def on_closing(self):
        if self.data_handler:
            self.data_handler.stop()  # Ensure the data handler stops
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()  # Stop the animation
        close_edf_writer()
        shutdown_external_apps()
        self.quit()
        self.destroy()

def start_data_handler(host, port):
    data_handler = DataHandler(host, port, packet_queue)
    app.data_handler = data_handler  # Assign data handler to the app for access in on_closing
    try:
        data_handler.start()
    except KeyboardInterrupt:
        data_handler.stop()


def launch_xtrodes_app():
    """Launch X-trodes PC App - Dev (UWP). No sleep — readiness is detected by wait_for_stable_stream."""
    print("[Startup] Launching X-trodes PC App - Dev...")
    subprocess.Popen(
        ['powershell', '-Command', f'Start-Process "shell:AppsFolder\\{XTRODES_APP_ID}"']
    )


def shutdown_external_apps():
    """Close the X-trodes PC App gracefully so it can finish writing its CSV before exiting."""
    subprocess.run(['taskkill', '/IM', 'BluetoothLEUniversal.exe'], capture_output=True)

    # Wait up to 15 seconds for the app to exit cleanly — only force-kill if it's still running
    for _ in range(15):
        time.sleep(1)
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq BluetoothLEUniversal.exe'],
            capture_output=True, text=True
        )
        if 'BluetoothLEUniversal.exe' not in result.stdout:
            print("[Shutdown] X-trodes app closed cleanly.")
            return
    print("[Shutdown] App did not close in time — force killing.")
    subprocess.run(['taskkill', '/F', '/IM', 'BluetoothLEUniversal.exe'], capture_output=True)


def wait_for_stable_stream(host, port, min_packets=3, stability_secs=1.0, timeout_secs=60):
    """Block until the X-trodes stream is flowing stably, then disconnect."""
    import socket
    deadline = time.time() + timeout_secs

    # Phase 1: poll the TCP port with lightweight probes until the app accepts connections.
    print("[Startup] Waiting for X-trodes app to be ready", end="", flush=True)
    while time.time() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                break
        except (socket.error, OSError):
            print(".", end="", flush=True)
            time.sleep(0.15)
    else:
        print("\n[Startup] Warning: app did not become ready within timeout — continuing anyway.")
        return False

    # Phase 2: connect DataHandler and wait for min_packets in stability_secs seconds.
    print("\n[Startup] App ready — checking data stream", end="", flush=True)
    monitor_queue = queue.Queue()
    handler = DataHandler(host, int(port), monitor_queue)
    threading.Thread(target=handler.start, daemon=True).start()

    received = 0
    stable_since = None
    try:
        while time.time() < deadline:
            try:
                monitor_queue.get(timeout=0.1)
                received += 1
                if stable_since is None:
                    stable_since = time.time()
                print(".", end="", flush=True)
            except queue.Empty:
                pass

            if (received >= min_packets
                    and stable_since is not None
                    and (time.time() - stable_since) >= stability_secs):
                print(f"\n[Startup] Stream stable — {received} packets received.")
                return True
    finally:
        handler.stop()

    print("\n[Startup] Warning: stream did not stabilize — continuing anyway.")
    return False


if __name__ == "__main__":
    # Step 1: Launch X-trodes PC App - Dev
    launch_xtrodes_app()

    # Step 2: Run checknetisolation in background — no need to block GUI startup
    bat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checknetisolation.bat")
    threading.Thread(target=lambda: subprocess.run([bat_path], shell=True), daemon=True).start()

    # Step 3: Launch main visualization app immediately
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = 1.0

    app = App()
    app.mainloop()

    # Ensure all threads are joined before exiting
    if app.data_handler:
        app.data_handler.stop()
