"""
this script is used to send the data in real time from the DAU to the C# script
that will control the virtual environment (Unity)
"""

from XtrRT.data import Data
import socket
import time
import keyboard
import numpy as np

if __name__ == '__main__':

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

    try:
        while True:
            print(data.exg_data.shape[0])
            if data.exg_data.shape[0] > 30:
                data_to_send = data.exg_data[-30:, :10]
                data_to_send = 5*np.mean(data_to_send, axis=0)  # Take the mean of the last 30 data points
                print("Sending data:", data_to_send)
                data_to_send = data_to_send.tobytes()  # Convert the data to bytes before sending
                connection.sendall(data_to_send)
            time.sleep(5/fs)  # Wait 5 data points before sending the next piece of data
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