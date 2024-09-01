"""
this script is used to send the offline data to the C# script
that will control the virtual environment (Unity)
"""

import socket
import time
import numpy as np
import pandas as pd

from CONSTS import mapping, blend_shapes

print(len(blend_shapes))


# function to fill the blendshapes data symmetrically to visualize full face movement
def fill_symetrical(data, mapping, blend_shapes):
    # Rename columns based on the mapping
    data.rename(columns=mapping, inplace=True)

    # Fill left columns with right columns values if the right columns exist
    for blend_shape in blend_shapes:
        # Reorder columns to match the specified order, adding missing columns with zeros
        if blend_shape not in data.columns:
            data[blend_shape] = 0
        if blend_shape.endswith('Left'):
            corresponding_right_column = blend_shape.replace('Left', 'Right')
            if corresponding_right_column in data.columns:  # actually all the blendshapes exist except for the CheeckRaiser
                data[blend_shape] = data[corresponding_right_column]

    # Keep only the desired columns and drop others
    data = data[blend_shapes]
    data = data.to_numpy()  # Convert dataframe to numpy array
    return data


def prepare_data(participant_number, session_number, model, avaraging_method, start_frame=4000):
    global mapping, blend_shapes

    # Load predicted AUs data
    # Read the CSV file with the first column as index and the first row as headers
    au_predicted_data = pd.read_csv(f"../data/participant_{participant_number}/S{session_number}/"
                                  f"participant_{participant_number}_S{session_number}_predicted_blendshapes_{model}.csv", index_col=0)
    # Start from the desired part
    au_predicted_data = au_predicted_data.iloc[start_frame:]

    # Load the full video data
    video_full_data = pd.read_csv(f"../data/participant_{participant_number}/S{session_number}/"
                                  f"participant_{participant_number}_S{session_number}_avatar_blendshapes_{avaraging_method}.csv", index_col=0)
    # Start from the desired part
    video_full_data = video_full_data.iloc[start_frame:]

    # Fill the data with the symmetrical values
    au_predicted_data = fill_symetrical(au_predicted_data, mapping, blend_shapes)
    video_full_data = fill_symetrical(video_full_data, mapping, blend_shapes)

    return au_predicted_data, video_full_data


if __name__ == '__main__':
    # choose the participant number, session number, and model (as strings)
    participant_number = '02'
    session_number = '2'
    model = "LR"  # "LR" for linear regression, "ETR" for extra trees regressor
    avatar_avaraging_method = "MEAN"

    # Prepare the data
    predicted_AUs, video_full_data = prepare_data(participant_number, session_number, model, avatar_avaraging_method)

    # Set up the first sender socket to send data1 to the first C# script
    sender_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sender_socket1.bind(('localhost', 65432))
    sender_socket1.listen(1)

    # Set up the second sender socket to send data2 to the second C# script
    sender_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sender_socket2.bind(('localhost', 65433))
    sender_socket2.listen(1)

    print("Waiting for connections...")
    connection1, client_address1 = sender_socket1.accept()
    print("Connected to:", client_address1)

    connection2, client_address2 = sender_socket2.accept()
    print("Connected to:", client_address2)
    try:
        for row1, row2 in zip(predicted_AUs, video_full_data):
            data_to_send1 = row1.tobytes()  # Convert data1 to bytes
            data_to_send2 = row2.tobytes()  # Convert data2 to bytes

            connection1.sendall(data_to_send1)
            connection2.sendall(data_to_send2)
            time.sleep(0.05)  # Wait 0.05 seconds before sending the next row
    finally:
        connection1.close()
        connection2.close()
        sender_socket1.close()
        sender_socket2.close()
