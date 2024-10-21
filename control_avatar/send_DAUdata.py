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
from send_data_to_CS import fill_symetrical, mapping, blend_shapes


def apply_and_order_ica_to_new_data(X_new, W, electrode_order):
    print("Applying ICA to new data")
    # Step 1: Apply the unmixing matrix to obtain the independent components
    Y_new = np.dot(W, X_new.T)

    # Order the data according to the electrode order
    Y_new_ordered = np.zeros_like(Y_new)
    for i, electrode in enumerate(electrode_order):
        if electrode != 16:
            Y_new_ordered[electrode, :] = Y_new[i, :]

    Y_normalized = normalize_ica_data(Y_new_ordered)

    return Y_normalized


def wavelet_denoising(emg_data_input, wavelet, level=5):
    signal = emg_data_input.copy()
    coefficients = pywt.wavedec(signal, wavelet, level)
    for j in range(1, len(coefficients)):
        coefficients[j] = pywt.threshold(coefficients[j], np.std(coefficients[j]))
    signal = pywt.waverec(coefficients, wavelet, level)
    return signal


def center(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = x - mean
    return centered, mean


def whiten(emg_signal):
    cov_matrix = covariance(emg_signal)
    U, S, V = np.linalg.svd(cov_matrix)
    d = np.diag(1.0 / np.sqrt(S))
    whiteM = np.dot(U, np.dot(d, U.T))
    emg_signal = np.dot(whiteM, emg_signal)
    return emg_signal


def calc_ica_components(emg_data):
    number_of_channels = emg_data.shape[0]
    K, W, Y = picard(emg_data, n_components=number_of_channels, ortho=True, extended=True, whiten=False,
                     max_iter=300)  # ICA algorithm
    return W

def image_load(image_path):
    # load the image, write the path where the image is saved (if there is no image uncomment these two lines)
    img = plt.imread(image_path)
    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = mpimg.imread(image_path)  # for heatmap

    # image dimensions
    height = img.shape[0]
    width = img.shape[1]

    return image, height, width

def norm(arr):
    myrange = np.nanmax(arr) - np.nanmin(arr)
    norm_arr = (arr - np.nanmin(arr)) / myrange
    return norm_arr

def plot_ica_heatmap(number_of_channels, W, height, width):
    # calculations for the heatmap
    inverse = np.absolute(inv(W))
    grid_y, grid_x = np.mgrid[1:height + 1, 1:width + 1]
    points = np.column_stack((x_coor, y_coor))
    f_interpolate = []
    for i in range(number_of_channels):
        interpolate_data = griddata(points, inverse[:, i], (grid_x, grid_y), method='linear')
        norm_arr = norm(interpolate_data)
        f_interpolate.append(norm_arr)  # plot heatmap
    return f_interpolate




def perform_ica_algorithm(emg_data, wavelet, height, width):
    emg_data = wavelet_denoising(emg_data, wavelet)
    emg_data, mean = center(emg_data)
    emg_data = whiten(emg_data)
    W = calc_ica_components(emg_data)
    f_interpolate = plot_ica_heatmap(emg_data.shape[0], W, height, width)
    return f_interpolate


def classify_participant_components_using_atlas(participant_data, centroids_lst, threshold, height, width):
    ica_electrode_order = [0 for i in range(16)]
    flags_list = [False for i in range(17)]
    min_dist_list = [0 for i in range(17)]
    # create a list for each participant with the channels that were assigned to each cluster
    for i in range(len(participant_data)):
        dist_from_centroids = []
        for centroid in centroids_lst[:-1]:
            centroid = centroid.reshape(height, width)
            dist = np.linalg.norm(participant_data[i, :] - centroid)
            dist_from_centroids.append(dist)
        # find distance from closest centroid (out of the 16 real clusters)
        closest_centroid_dist = np.min(dist_from_centroids)
        closest_centroid_index = np.argmin(dist_from_centroids)
        if closest_centroid_dist < threshold:
            if not flags_list[closest_centroid_index]:
                flags_list[closest_centroid_index] = True
                min_dist_list[closest_centroid_index] = closest_centroid_dist
                ica_electrode_order[i] = closest_centroid_index
            else:
                if min_dist_list[closest_centroid_index] > closest_centroid_dist:
                    # get the index of the channel that was already assigned to the cluster
                    index = ica_electrode_order.index(closest_centroid_index)
                    ica_electrode_order[index] = 16
                    ica_electrode_order[i] = closest_centroid_index
                    min_dist_list[closest_centroid_index] = closest_centroid_dist
                else:
                    # if the channel was already assigned to a cluster, assign it to the garbage cluster
                    ica_electrode_order[i] = 16
        else:
            # if the channel was already assigned to a cluster, assign it to the garbage cluster
            ica_electrode_order[i] = 16
    ica_after_order = np.zeros_like(participant_data)
    for i in range(16):
        if int(ica_electrode_order[i]) != 16:
            ica_after_order[ica_electrode_order[i], :] = participant_data[i, :]
    return ica_after_order


def run_ica(data, fs, wavelet, height, width, centroids_lst, threshold):
    f_interpolate = perform_ica_algorithm(data, fs, wavelet, height, width)
    ica_after_order = classify_participant_components_using_atlas(f_interpolate, centroids_lst, threshold, height, width)
    return ica_after_order

def filter_signal(data, fs):
    # apply notch filter to remove 50Hz noise
    b, a = signal.iirnotch(50, 30, fs)
    data = signal.filtfilt(b, a, data, axis=1)
    # apply bandpass filter to remove high frequency noise
    b, a = signal.butter(4, [35, 249], fs=fs, btype='band')
    filtered_signal = signal.filtfilt(b, a, data, axis=1)
    # add notch filter of 200 hz
    b, a = signal.iirnotch(200, 30, fs)
    filtered_signal = signal.filtfilt(b, a, filtered_signal)
    # add notch filter of 100 hz
    b, a = signal.iirnotch(100, 30, fs)
    filtered_signal = signal.filtfilt(b, a, filtered_signal)
    return filtered_signal

def process_emg_data(data, fs, wavelet, height, width, centroids_lst, threshold, model):
    print("running function")
    # filter the data
    data = filter_signal(data, fs)
    # Apply ICA transformation to the new data
    ica_after_order = run_ica(data, fs, wavelet, height, width, centroids_lst, threshold)
    # apply RMS averaging
    Y_data = np.sqrt(np.mean(np.power(ica_after_order, 2)))
    print("Y data:", Y_data)

    # Pass through ML model for prediction
    model_prediction = model.predict(Y_data.T)
    print("Predictions:", model_prediction)
    model_prediction = pd.DataFrame(predictions, columns=relevant_blendshapes)

    # Fill the symmetrical blendshapes
    full_predictions = fill_symetrical(model_prediction, mapping, blend_shapes)

    return full_predictions


if __name__ == '__main__':
    # choose the participant number, session number, and model (as strings)
    participant_number = '02'
    session_number = '2'
    model_name = "ImprovedEnhancedTransformNet"  # "LR" for linear regression, "ETR" for extra trees regressor
    wavelet = "db15"


    script_dir = r"C:\Users\YH006_new\fEMG_to_avatar\data_process"
    # Define the project folder by going up two levels from the script's directory
    project_folder = r"C:\Users\YH006_new\fEMG_to_avatar"
    # Load pre-trained ML model
    model = joblib.load(fr"{project_folder}/data/participant_{participant_number}/S{session_number}/"
                           f"participant_{participant_number}_S{session_number}_blendshapes_{model_name}.joblib")


    atlas_folder = os.path.join(script_dir, 'atlas')
    image_path = os.path.join(project_folder, 'side.jpg')  # path to the image of the face to show the heatmaps on
    x_coor_path = os.path.join(atlas_folder, 'side_x_coor.npy')
    y_coor_path = os.path.join(atlas_folder, 'side_y_coor.npy')
    x_coor = np.load(x_coor_path)
    y_coor = np.load(y_coor_path)
    # load picture
    image, height, width = image_load(image_path)
    n = 17
    print("Loading the centroids...")
    threshold = np.load(f"{atlas_folder}/threshold.npy")
    # load the centroids
    centroids_lst = []
    for i in range(0, n):
        current_centroid = np.load(f"{atlas_folder}/cluster_{i + 1}.npy")
        current_centroid = np.nan_to_num(current_centroid, nan=0)
        centroids_lst.append(current_centroid)
    print("Centroids loaded")

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
            print(data.exg_data.shape[0])
            if data.exg_data.shape[0] > 2 * fs / sending_data_frequency:
                print("enough data to send")
                windowed_data = data.exg_data[-int(2 * fs / sending_data_frequency):,
                                :16]  # send 2 windows of 0.05 seconds every time
                # print(windowed_data)
                data_to_send = process_emg_data(windowed_data, fs, wavelet, height, width, centroids_lst, threshold, model)  # Process the data
                print("Sending data:", data_to_send.shape)
                data_to_send = data_to_send.tobytes()  # Convert the data to bytes before sending
                connection.sendall(data_to_send)
            time.sleep(1 / sending_data_frequency)  # Wait the desired time according to the sending frequency
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
