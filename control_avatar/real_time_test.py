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


def listen_for_ble_data(host, port, data_callback):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Listening for BLE data on {host}:{port}")

    while True:
        client_socket, address = server_socket.accept()
        print(f"Connected to BLE device at {address}")

        try:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                data_callback(data)
        except Exception as e:
            print(f"Error receiving BLE data: {e}")
        finally:
            client_socket.close()


def setup_server_for_csharp(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Waiting for C# connection on {host}:{port}")
    connection, address = server_socket.accept()
    print(f"Connected to C# at {address}")
    return connection


def process_and_send_data(raw_data, csharp_connection, data_object, W, electrode_order, ml_model, fs):
    # Add the new data to the Data object
    data_object.add_data(raw_data)

    if data_object.exg_data is not None and data_object.exg_data.shape[0] > 2 * fs / sending_data_frequency:
        windowed_data = data_object.exg_data[-int(2 * fs / sending_data_frequency):, :16]
        data_to_send = process_emg_data(windowed_data, W, electrode_order, ml_model, fs)
        print("Sending data:", data_to_send.shape)
        data_to_send = data_to_send.tobytes()
        csharp_connection.sendall(data_to_send)


ble_host, ble_port = 'localhost', 20001
csharp_host, csharp_port = 'localhost', 65432
global sending_data_frequency
sending_data_frequency = 20  # 20 Hz

try:
    csharp_connection = setup_server_for_csharp(csharp_host, csharp_port)

    # Initialize the Data object
    data_object = Data(csharp_host, csharp_port, verbose=False, timeout_secs=15, save_as="test.edf")
    data_object.start()

    fs = int(data_object.fs_exg)  # Sampling frequency of the exg data

    # Start listening for BLE data in a separate thread
    ble_thread = threading.Thread(target=listen_for_ble_data, args=(ble_host, ble_port,
                                                                    lambda data: process_and_send_data(data,
                                                                                                       csharp_connection,
                                                                                                       data_object, W,
                                                                                                       electrode_order,
                                                                                                       ml_model, fs)))
    ble_thread.start()

    # Main thread can handle other tasks or simply wait
    ble_thread.join()

except KeyboardInterrupt:
    print("Process terminated by user")
finally:
    if 'data_object' in locals():
        data_object.stop()
    if 'csharp_connection' in locals():
        csharp_connection.close()
    print("Process terminated")
