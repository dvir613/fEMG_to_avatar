import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider
import mne

def offset_calc(data, sampling_rate):
    # max_model = np.mean(data[:, 30 * sampling_rate:-30 * sampling_rate], axis=1) + 4*np.std(data[:, 30 * sampling_rate:-30 * sampling_rate], axis=1)
    # max_model = np.roll(max_model, 1)
    offsets = (6*np.std(data, axis=1))
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    for offset in range(1, len(offsets)):
        offsets[offset] += 0.2 * offset
    return offsets

def ica_time_plot(signals, annotations_bool, sampling_rate, annotations, annotation_times, T_plots):
    ICA_fig, ICA_axs = plt.subplots(figsize=(16, 8))
    ICA_axs.set_title('Signals')

    # plot_colormap = plt.get_cmap("Set2")
    # ICA_axs.set_xlabel('time [min:sec]')
    # time_func = (lambda a: divmod(int(a / sampling_rate), 60))
    # ticks = ticker.FuncFormatter(lambda x, pos: (str(time_func(x)[0]).zfill(2) + ':' + str(time_func(x)[1]).zfill(2)))
    # ICA_axs.xaxis.set_major_formatter(ticks)

    if annotations_bool:
        # annotation_descriptions = [annot['description'] for annot in annotations]
        annotation_descriptions = [annot['description'].split('_', 1)[1] for annot in annotations]
        unique_annotations, reverse_indexes = np.unique(annotation_descriptions, return_inverse=True)
        # unique_colors = cm.rainbow(np.linspace(0, 1, unique_annotations.shape[0]))
        # for time, annotation, color in zip(annotation_times, unique_annotations[reverse_indexes], unique_colors[reverse_indexes]):
        #     ICA_axs.axvline(x=time * sampling_rate, linestyle='--', label=annotation, color=color, linewidth=1.75)
        for time in annotation_times:
            ICA_axs.axvline(x=time * sampling_rate, linestyle='--', color='black', linewidth=1.75)

        # ICA_axs.legend()
        # Set x-ticks at annotation times
        annotation_times_ticks = [time * sampling_rate for time in annotation_times]
        ICA_axs.set_xticks(annotation_times_ticks)
        ICA_axs.set_xticklabels(annotation_descriptions, rotation=45, fontsize=10, color='black')

    new_signals = signals / (np.max(np.abs(signals)) + 3*np.std(signals)) - 1
    for source in range(signals.shape[0]):
        new_signals[source] = np.clip(new_signals[source], np.mean(new_signals[source]) - 7 * np.std(new_signals[source]), np.mean(new_signals[source]) + 7 * np.std(new_signals[source]))
    # new_signals = new_signals / 15

    offsets = offset_calc(new_signals, sampling_rate)
    ICA_axs.margins(0)
    # ICA_axs.plot((new_signals[:, :T_plots] + offsets[:, np.newaxis]).T, color=plot_colormap.colors[0])
    ICA_axs.plot((new_signals[:, :T_plots] + offsets[:, np.newaxis]).T, color='darkblue')

    y_values = (new_signals[:, :T_plots] + offsets[:, np.newaxis]).mean(axis=1)
    y_labels = [f'Channel {i+1}' for i in range(signals.shape[0])]
    plt.yticks(y_values, y_labels)

    # colors = [plot_colormap.colors[0] for _ in range(signals.shape[0])]
    # for ytick, color in zip(ICA_axs.get_yticklabels(), colors):
    #     ytick.set_color(color)
    for ytick in ICA_axs.get_yticklabels():
        ytick.set_color('darkblue')

    ICA_fig.tight_layout()

    ICA_fig.canvas.draw()
    ICA_fig.canvas.flush_events()
    plt.show()


# determine whether to plot ICA or EMG data
plot_ICA = True
plot_EMG = False

# Load the EDF file
edf_path = r"..\data\participant_01\participant_01.edf"
ICA_path = r"../data/participant_01/participant_01_db15_Y.npy"
ICA_order_path = r"../data/participant_01/participant_01_db15_electrode_order.npy"

if plot_ICA:
    emg_file = mne.io.read_raw_edf(edf_path, preload=False)
if plot_EMG:
    emg_file = mne.io.read_raw_edf(edf_path, preload=True)
emg_fs = int(emg_file.info['sfreq'])  # sampling frequency of the emg data

if plot_EMG:
    # Extract signal data
    n_channels = 16
    signals = np.zeros((n_channels, emg_file.get_data().shape[1] - 110*emg_fs))  # remove the first 120 seconds (nothing interesting there)

    for i in range(n_channels):
        # first: pass all channels through a notch filter of 50 Hz (powerline interface)
        signals[i, :] = mne.filter.notch_filter(emg_file.get_data()[i, 110*emg_fs:], emg_fs, freqs=50, trans_bandwidth=1,
                                                method='fir', copy=False)
        signals[i, :] = mne.filter.filter_data(emg_file.get_data()[i, 110*emg_fs:], emg_fs, l_freq=30, h_freq=249, method='fir', copy=True)

if plot_ICA:
    ica_before_order = np.load(ICA_path)
    electrode_order = np.load(ICA_order_path)
    signals = np.zeros((ica_before_order.shape[0], ica_before_order.shape[1] - 110*emg_fs))
    for i in range(16):
        if int(electrode_order[i]) != 16:
            signals[electrode_order[i], :] = ica_before_order[i, 110*emg_fs:]

# Load the annotations
edf_annotations = emg_file.annotations
events_annotations = ['05_Forehead', '07_Eye_gentle', '09_Eye_tight', '12_Nose', '14_Smile_closed', '16_Smile_open',
                      '19_Lip_pucker', '21_Cheeks', '23_Snarl', '26_Depress_lip', '36_Natural_smile']
event_annotations = [annot for annot in edf_annotations if annot['description'] in events_annotations]
event_timings = [annot['onset'] - 110 for annot in event_annotations]
print(event_timings)
annotations_bool = True

T_plots = signals.shape[1]  # plot all data

ica_time_plot(signals, annotations_bool, emg_fs, event_annotations, event_timings, T_plots)

