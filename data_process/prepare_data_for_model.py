import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timezone, timedelta
# set np seed
np.random.seed(42)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the project folder by going up two levels from the script's directory
project_folder = os.path.abspath(os.path.join(script_dir, '..'))


# window length and step length in seconds
def sliding_window(data, method, fs, window_length=0.1, step_length=0.05):
    window_size = int(window_length * fs)
    step_size = int(step_length * fs)
    # Calculate number of windows
    num_windows = (data.shape[1] - window_size) // step_size + 1
    # print(num_windows)
    result = np.zeros((data.shape[0], num_windows))
    if method == "RMS":
        data = data ** 2
        for i in range(data.shape[0]):
            for j in range(num_windows):
                result[i, j] = np.sqrt(np.mean(data[i, j * step_size:j * step_size + window_size]))
    if method == "MEAN":
        for i in range(data.shape[0]):
            for j in range(num_windows):
                result[i, j] = np.mean(data[i, j * step_size:j * step_size + window_size])
    if method == "Max":
        for i in range(data.shape[0]):
            for j in range(num_windows):
                result[i, j] = np.max(data[i, j * step_size:j * step_size + window_size])

    return result


def extract_and_order_ica_data(participant_ID, session_folder_path, session_number):
    ica_before_order= None
    electrode_order = None
    for file in os.listdir(session_folder_path):
        if file == fr"{participant_ID}_{session_number}_db15_Y.npy":
            ica_before_order = np.load(f"{session_folder_path}/{file}")
        if file == fr"{participant_ID}_{session_number}_db15_electrode_order.npy":
            electrode_order = np.load(f"{session_folder_path}/{file}")
        continue
    if ica_before_order.any() and electrode_order.any():
        ica_after_order = np.zeros_like(ica_before_order, dtype=float)
        for i in range(16):
            if int(electrode_order[i]) != 16:
                ica_after_order[electrode_order[i], :] = ica_before_order[i, :]

        return ica_after_order
    else:
        print("Error: ICA data or electrode order not found")
        return None


def normalize_ica_data(ica_data):
    # Normalize the data to include 99% of the data in the range [-1, 1] (avoiding outliers)
    ica_data = ica_data / (np.mean(np.abs(ica_data), axis=1).reshape(-1, 1)
                           + 4 * np.std(np.abs(ica_data), axis=1).reshape(-1, 1) + 1e-7)
    # added epsilon to avoid division by zero
    return ica_data


def get_annotations_timings(emg_file, annotations_list=None):
    if annotations_list is None:
        annotations_list = ['05_Forehead', '07_Eye_gentle', '09_Eye_tight', '12_Nose', '14_Smile_closed',
                            '16_Smile_open', '19_Lip_pucker', '21_Cheeks', '23_Snarl', '26_Depress_lip']
    edf_annotations = emg_file.annotations
    events_annotations = [annot for annot in edf_annotations if annot['description'] in annotations_list]
    events_timings = [annot['onset'] for annot in events_annotations]
    return events_timings


def prepare_relevant_data(data, emg_file, fs, trials_lst_timing, events_timings=None, segments_length=35, norm=None, averaging="RMS"):
    fs = int(fs)
    if events_timings is None:
        events_timings = get_annotations_timings(emg_file)
    # split to train and test: first 2 repetitions are for training, the third is for testing
    relevant_data_train = []
    relevant_data_test = []
    for i in range(len(trials_lst_timing)):
        if i % 3 == 0 and i!=0:
            relevant_data_test.append(data[:,int(trials_lst_timing[i][0]) * fs:int(trials_lst_timing[i][1]) * fs])
        else:
            relevant_data_train.append(data[:,int(trials_lst_timing[i][0] )* fs:int(trials_lst_timing[i][1]) * fs])
    relevant_data_train = np.concatenate(relevant_data_train, axis=1)
    relevant_data_test = np.concatenate(relevant_data_test, axis=1)
    if norm == "ICA":
        relevant_data_train = normalize_ica_data(relevant_data_train)
        relevant_data_test = normalize_ica_data(relevant_data_test)
    # sliding window averaging
    relevant_data_train = sliding_window(relevant_data_train, method=averaging, fs=fs)
    relevant_data_test = sliding_window(relevant_data_test, method=averaging, fs=fs)
    return relevant_data_train, relevant_data_test


def prepare_relevant_data_new(data, emg_file, fs, trials_lst_timing, rand_lst=None, events_timings=None,
                              segments_length=35, norm=None, averaging="RMS"):
    data_is_ICA = False
    fs = int(fs)
    relevant_data_train = []
    relevant_data_test = []

    if rand_lst is None:
        rand_lst = []
        data_is_ICA = True

    for i in range(0, len(trials_lst_timing), 3):
        group = trials_lst_timing[i:i + 3]
        if group:  # Check if the group is not empty
            if data_is_ICA:
                rand = np.random.choice(len(group))
                rand_lst.append(i + rand)
            else:
                rand = rand_lst[i // 3] - i  # Adjust the rand index to be relative to the current group

            selected_trial = group[rand]
            relevant_data_test.append(data[:, int(selected_trial[0]) * fs:int(selected_trial[1]) * fs])

            for j, trial in enumerate(group):
                if j != rand:
                    relevant_data_train.append(data[:, int(trial[0]) * fs:int(trial[1]) * fs])

    relevant_data_train = np.concatenate(relevant_data_train, axis=1)
    relevant_data_test = np.concatenate(relevant_data_test, axis=1)

    if norm == "ICA":
        relevant_data_train = normalize_ica_data(relevant_data_train)
        relevant_data_test = normalize_ica_data(relevant_data_test)

    # sliding window averaging
    relevant_data_train = sliding_window(relevant_data_train, method=averaging, fs=fs)
    relevant_data_test = sliding_window(relevant_data_test, method=averaging, fs=fs)

    return relevant_data_train, relevant_data_test, rand_lst


def plot_ica_vs_blendshapes(avatar_data, blendshapes, emg_file, emg_fs, events_timings, ica_after_order, participant_ID):
    emg_fs = int(emg_fs)
    rands_lst = []
    relevant_data_test = []
    for i in range(0, len(events_timings), 3):
        group = events_timings[i:i + 3]
        if group:  # Check if the group is not empty
            rand = np.random.choice(len(group))
            selected_event = group[rand]
            relevant_data_test.append(
                ica_after_order[:, int(selected_event[0]) * emg_fs:int(selected_event[1]) * emg_fs])
            rands_lst.append(i + rand)
    relevant_data_train_emg = np.concatenate(relevant_data_test, axis=1)
    time_delta = get_time_delta(emg_file, avatar_data, participant_ID)
    avatar_data = avatar_data.to_numpy().T
    print("original avatar data shape: ", avatar_data.shape)
    avatar_fs = 60
    frames_to_cut = int(time_delta * avatar_fs)
    avatar_data = avatar_data[:, frames_to_cut:]
    print("avatar data cut shape: ", avatar_data.shape)
    relevant_data_avatar_test = []
    for i, rand_index in enumerate(rands_lst):
        selected_event = events_timings[rand_index]
        relevant_data_avatar_test.append(
            avatar_data[:, int(selected_event[0]) * avatar_fs:int(selected_event[1]) * avatar_fs])
    relevant_data_train_avatar = np.concatenate(relevant_data_avatar_test, axis=1)
    # Define blendshapes to exclude
    exclude_blendshapes = {'EyeLookInRight', 'NoseSneerRight', 'EyeLookUpRight', 'MouthDimpleRight'}

    # Filter blendshapes to include only those with 'Right' and not in the exclude list
    right_blendshapes = [bs for bs in blendshapes if
                         'Right' in bs and bs not in exclude_blendshapes]
    right_indices = [i for i, bs in enumerate(blendshapes) if
                     'Right' in bs and bs not in exclude_blendshapes]

    fig, axs = plt.subplots(len(right_blendshapes), 2, figsize=(24, 24), dpi=300)
    plt.rcParams.update({'font.size': 28})  # Increase base font size

    # Normalize ICA and blendshape data
    def normalize_data(data):
        return data / np.max(data)

    normalized_emg = [normalize_data(data) for data in relevant_data_train_emg]
    normalized_avatar = [normalize_data(data) for data in relevant_data_train_avatar]

    # Calculate the overall time range for both EMG and avatar data
    max_time_emg = max(len(data) for data in normalized_emg) / emg_fs
    max_time_avatar = max(len(data) for data in normalized_avatar) / avatar_fs
    max_time = max(max_time_emg, max_time_avatar)

    for i, (blendshape, original_index) in enumerate(zip(right_blendshapes, right_indices)):
        # Reverse the order of ICA plots
        ica_index = len(right_blendshapes) - i - 1

        if ica_index < len(normalized_emg):
            time_axis = np.arange(len(normalized_emg[ica_index])) / emg_fs
            axs[i, 0].plot(time_axis, normalized_emg[ica_index], linewidth=0.8)
            axs[i, 0].set_ylabel(f'ICA {ica_index + 1}', rotation=0, ha='right', va='center', size=28)

        time_axis = np.arange(len(normalized_avatar[original_index])) / avatar_fs
        axs[i, 1].plot(time_axis, normalized_avatar[original_index], linewidth=0.8)
        axs[i, 1].set_ylabel(blendshape, rotation=0, ha='right', va='center', size=28)

        # Set the same x-axis limits for both subplots
        axs[i, 0].set_xlim(0, max_time)
        axs[i, 1].set_xlim(0, max_time)

        # Remove top and right spines, keep only y-axis
        for ax in [axs[i, 0], axs[i, 1]]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labelsize=20)
            ax.tick_params(axis='y', which='both', labelsize=18)  # Make yticks smaller

        # Add horizontal line at y=0 for all subplots
        axs[i, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        axs[i, 1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # Remove x-axis labels for all but the bottom subplot
        if i < len(right_blendshapes) - 1:
            axs[i, 0].set_xticklabels([])
            axs[i, 1].set_xticklabels([])
        else:
            # Add x-axis only for the bottom subplots
            for ax in [axs[i, 0], axs[i, 1]]:
                ax.spines['bottom'].set_visible(True)
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
            axs[i, 0].set_xlabel('Time (s)', size=28)
            axs[i, 1].set_xlabel('Time (s)', size=28)

    # Add ylabels for the 8 subplots
    axs[0, 0].set_title('ICA Components', size=28)
    axs[0, 1].set_title('Blendshapes', size=28)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect parameter to make room for suptitle
    plt.savefig(fr"{project_folder}\results\{participant_ID}_ICA_vs_blendshapes.png")
    plt.close()

def prepare_avatar_relevant_data(participant_ID, avatar_data, emg_file, relevant_data_train_emg, relevant_data_test_emg, trials_lst_timing, rand_lst, fs=60, events_timings=None, segments_length=35, norm=None, averaging="MEAN"):
    time_delta = get_time_delta(emg_file, avatar_data, participant_ID)
    avatar_data = avatar_data.to_numpy().T
    fs_emg = emg_file.info['sfreq']
    print("original avatar data shape: ", avatar_data.shape)
    avatar_fps = fs  # data collection was in 60 fps
    frames_to_cut = int(time_delta * avatar_fps)
    avatar_data = avatar_data[:, frames_to_cut:]
    print("avatar data cut shape: ", avatar_data.shape)

    relevant_data_train_avatar, relevant_data_test_avatar, rand_lst =  prepare_relevant_data_new(avatar_data, emg_file, fs, trials_lst_timing, rand_lst, events_timings=events_timings,
                                                           segments_length=segments_length, norm=norm, averaging=averaging)
    return relevant_data_train_avatar, relevant_data_test_avatar


def get_time_delta(emg_file, avatar_file, participant_ID):
    # get edf start time
    if participant_ID == 'participant_01':
        edf_start_time = emg_file.info['meas_date']
        edf_start_time = edf_start_time.time()
    else:
        # the last annotation (before "stop recording") is: "data.start_time: YYYY-MM-DD HH:MM:SS.fff"
        edf_annotations = emg_file.annotations
        edf_start_time = edf_annotations[-2]['description'].split(' ')[-1]
        edf_start_time = datetime.strptime(edf_start_time, "%H:%M:%S.%f")
        edf_start_time = edf_start_time.time()

    # get avatar start time
    avatar_start_time = avatar_file.index[0]
    avatar_start_time = datetime.strptime(avatar_start_time, "%H:%M:%S.%f")
    avatar_start_time = avatar_start_time.time()

    # Calculate the time delta ignoring the date
    edf_start_seconds = timedelta(hours=edf_start_time.hour,
                                  minutes=edf_start_time.minute,
                                  seconds=edf_start_time.second,
                                  microseconds=edf_start_time.microsecond).total_seconds()

    avatar_start_seconds = timedelta(hours=avatar_start_time.hour,
                                     minutes=avatar_start_time.minute,
                                     seconds=avatar_start_time.second,
                                     microseconds=avatar_start_time.microsecond).total_seconds()

    time_delta = edf_start_seconds - avatar_start_seconds
    print("time difference: ", int(time_delta * 1000), "milliseconds")
    return time_delta


def add_eeg(emg_file, emg_relevant_data_averaged, fs, events_timings, segments_length=35, norm="ICA", averaging="RMS"):
    eeg = np.zeros((2, emg_relevant_data_averaged.shape[1]))
    eeg[0, :] = mne.filter.filter_data(emg_file.get_data()[14], fs, l_freq=0.3, h_freq=30, method='fir', copy=True)
    eeg[1, :] = mne.filter.filter_data(emg_file.get_data()[15], fs, l_freq=0.3, h_freq=30, method='fir', copy=True)
    # get relevant normalized averaged data
    eeg_relevant_data_averaged = prepare_relevant_data(eeg, emg_file=emg_file, fs=fs, events_timings=events_timings,
                                                       segments_length=35, norm="ICA", averaging="RMS")
    # concatenate
    relevant_data = np.concatenate((emg_relevant_data_averaged, eeg_relevant_data_averaged), axis=0)
    return relevant_data





