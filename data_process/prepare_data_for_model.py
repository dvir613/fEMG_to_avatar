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
def sliding_window(data, method, fs, window_length=0.1):
    window_size = int(window_length * fs)

    # Calculate number of windows
    num_windows = data.shape[1] // window_size

    # print(num_windows)
    result = np.zeros((data.shape[0], num_windows))
    if method == "RMS":
        for i in range(data.shape[0]):
            for j in range(num_windows):
                result[i, j] = np.sqrt(np.mean(np.power(data[i, j * window_size:j * window_size + window_size], 2)))
    if method == "MEAN":
        for i in range(data.shape[0]):
            for j in range(num_windows):
                result[i, j] = np.mean(data[i, j * window_size:j * window_size + window_size])
    if method == "Max":
        for i in range(data.shape[0]):
            for j in range(num_windows):
                result[i, j] = np.max(data[i, j * window_size:j * window_size + window_size])

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


def prepare_relevant_data_new(data, emg_file, fs, trials_lst_timing, for_plot_flag=False, rand_lst=None, events_timings=None,
                              segments_length=35, norm=None, averaging="RMS"):
    data_is_ICA = False
    fs = int(fs)
    relevant_data_train = []
    relevant_data_test = []
    test_data_timing = []

    if rand_lst is None:
        num_groups = len(trials_lst_timing) // 3
        rand_lst = np.random.choice(3, size=num_groups)  # Adjust rand_lst to be relative to each group

    for i, rand in enumerate(rand_lst):
        group_start = i * 3
        group = trials_lst_timing[group_start:group_start + 3]
        if group:  # Check if the group is not empty
            selected_trial = group[rand]
            current_facial_expressions = data[:, int(selected_trial[0]) * fs:int(selected_trial[1]) * fs]
            if for_plot_flag:
                relevant_data_test.append(current_facial_expressions)
                test_data_timing.append(selected_trial)
            else:
                relevant_data_test.append(sliding_window(current_facial_expressions, method=averaging, fs=fs))
                test_data_timing.append(selected_trial)

            for j, trial in enumerate(group):
                if j != rand:
                    current_facial_expressions = data[:, int(trial[0]) * fs:int(trial[1]) * fs]
                    if for_plot_flag:
                        relevant_data_train.append(current_facial_expressions)
                    else:
                        relevant_data_train.append(sliding_window(current_facial_expressions, method=averaging, fs=fs))

    relevant_data_train = np.concatenate(relevant_data_train, axis=1)
    relevant_data_test = np.concatenate(relevant_data_test, axis=1)

    return relevant_data_train, relevant_data_test, rand_lst, test_data_timing


def prepare_avatar_relevant_data(participant_ID, avatar_data, emg_file, relevant_data_train_emg, relevant_data_test_emg, trials_lst_timing, for_plot_flag, rand_lst, fs=60, events_timings=None, segments_length=35, norm=None, averaging="MEAN"):
    time_delta = get_time_delta(emg_file, avatar_data, participant_ID)
    avatar_data = avatar_data.to_numpy().T
    print("original avatar data shape: ", avatar_data.shape)
    frames_to_cut = int(time_delta * fs)
    avatar_data = avatar_data[:, frames_to_cut:]
    print("avatar data cut shape: ", avatar_data.shape)

    relevant_data_train_avatar, relevant_data_test_avatar, rand_lst ,test_data_timing =  prepare_relevant_data_new(avatar_data, emg_file, fs, trials_lst_timing, for_plot_flag, rand_lst, events_timings=events_timings,
                                                           segments_length=segments_length, norm=norm, averaging=averaging)
    # save only the same trials as the emg data
    relevant_data_train_avatar = relevant_data_train_avatar[:, :relevant_data_train_emg.shape[1]]
    relevant_data_test_avatar = relevant_data_test_avatar[:, :relevant_data_test_emg.shape[1]]
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





