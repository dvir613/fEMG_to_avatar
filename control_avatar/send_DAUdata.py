"""
this script is used to send the data in real time from the DAU to the C# script
that will control the virtual environment (Unity)
the EMG data acquisition should be in 500Hz (the same as in the learning)
"""
import pandas as pd

from XtrRT.data import Data
import socket
import time
import keyboard
import numpy as np
import joblib

from data_process.prepare_data_for_model import normalize_ica_data
from CONSTS import *
from send_data_to_CS import fill_symetrical
from data_process.classifying_ica_components import filter_signal, wavelet_denoising, center, whiten


def preprocess_emg_data(emg_data_chunk, fs=500, wavelet='db15'):
    # Hila's preprocess functions should be here
    # filter the data
    filtered_emg_data_chunk = filter_signal(emg_data_chunk, fs)
    # wavelet denoising
    # TODO: verify with Hila the best parameters maybe need to change the wavelet denoising function for real-time
    # filtered_emg_data_chunk = wavelet_denoising(filtered_emg_data_chunk, fs, wavelet, window_size=0.1, level=5)
    # center the data
    filtered_emg_data_chunk, _ = center(filtered_emg_data_chunk)
    # whiten the data
    filtered_emg_data_chunk = whiten(filtered_emg_data_chunk)
    return filtered_emg_data_chunk


def apply_and_order_ica_to_new_data(X_new, W, electrode_order):
    # Step 1: Apply Hila's preprocess functions
    X_new = preprocess_emg_data(X_new)

    # Step 2: Apply the unmixing matrix to obtain the independent components
    Y_new = np.dot(W, X_new.T)

    # Step 3: Order the data according to the electrode order
    Y_new_ordered = np.zeros_like(Y_new)
    for i, electrode in enumerate(electrode_order):
        if electrode != 16:
            Y_new_ordered[electrode, :] = Y_new[i, :]

    # Step 4: Normalize the data
    Y_normalized = normalize_ica_data(Y_new_ordered)

    return Y_normalized


def process_emg_data(emg_data_chunk, W, electrode_order, ml_model, fs=500):
    # filter the data
    emg_data_chunk = preprocess_emg_data(emg_data_chunk, fs)
    # Apply ICA transformation to the new data
    Y_data = apply_and_order_ica_to_new_data(emg_data_chunk, W, electrode_order)

    # apply RMS averaging
    Y_data = np.sqrt(np.mean(Y_data**2, axis=1))

    # Pass through ML model for prediction
    predictions = ml_model.predict(Y_data.reshape(1, -1))

    # Fill the symmetrical blendshapes
    predictions = pd.DataFrame(predictions, columns=relevant_blendshapes)
    full_predictions = fill_symetrical(predictions, mapping, blend_shapes)

    return full_predictions


if __name__ == '__main__':
    # choose the participant number, session number, and model (as strings)
    participant_number = '02'
    session_number = '1'
    model = "LR"  # "LR" for linear regression, "ETR" for extra trees regressor
    wavelet = "db15"

    # Load pre-trained ICA transformer
    W = np.load(f"../data/participant_{participant_number}/S{session_number}/"
                f"participant_{participant_number}_S{session_number}_{wavelet}_W.npy")
    electrode_order = np.load(f"../data/participant_{participant_number}/S{session_number}/"
                              f"participant_{participant_number}_S{session_number}_{wavelet}_electrode_order.npy")

    # Load pre-trained ML model
    ml_model = joblib.load(f"../data/participant_{participant_number}/S{session_number}/"
                           f"participant_{participant_number}_S{session_number}_blendshapes_model_{model}.joblib")

    sending_data_frequency = 20  # 20 Hz/fps = 50 ms

    host_name = "127.0.0.1"
    port = 20001
    n_bytes = 1024
    data = Data(host_name, port, verbose=False, timeout_secs=15, save_as="test.edf")
    data.start()

    data.add_annotation("Start recording")

    time.sleep(5)  # Record for 5 seconds
    fs = int(data.fs_exg)  # Sampling frequency of the exg data

    # Set up the sender socket to send tha data to C# script that will control unity
    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sender_socket.bind(('localhost', 65432))
    sender_socket.listen(1)

    print("Waiting for a connection...")
    connection, client_address = sender_socket.accept()
    print("Connected to:", client_address)

    # just a dummy example to execute some preprocess before sending the data
    try:
        while True:
            if data.exg_data.shape[0] > 2*fs/sending_data_frequency:
                windowed_data = data.exg_data[-int(2*fs/sending_data_frequency):, :16]  # send 2 windows of 0.05 seconds every time
                data_to_send = process_emg_data(windowed_data, W, electrode_order, ml_model, fs)  # Process the data
                print("Sending data:", data_to_send.shape)
                data_to_send = data_to_send.tobytes()  # Convert the data to bytes before sending
                connection.sendall(data_to_send)
            time.sleep(1/sending_data_frequency)  # Wait the desired time according to the sending frequency
    finally:
        connection.close()
        sender_socket.close()

        print("Press ESC to finish the running of the code")
        # This loop will keep running until the ESC key is pressed
        while True:
            if keyboard.is_pressed('esc'):
                break

        data.add_annotation("Stop recording")
        data.stop()

        print(data.annotations)
        print('process terminated')