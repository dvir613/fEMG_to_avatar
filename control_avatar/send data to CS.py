"""
this script is used to send the offline data to the C# script
that will control the virtual environment (Unity)
"""

import socket
import time
import numpy as np
import pandas as pd


mapping = {
    "BrowInnerUp": "BrowInnerUpRight",
    "CheekPuff": "CheekPuffRight",
}


blend_shapes = [
    "BrowDownLeft",
    "BrowDownRight",
    "BrowInnerUpLeft",
    "BrowInnerUpRight",
    "BrowOuterUpLeft",
    "BrowOuterUpRight",
    "CheekPuffLeft",
    "CheekPuffRight",
    "CheekRaiserLeft",
    "CheekRaiserRight",
    "CheekSquintLeft",
    "CheekSquintRight",
    "EyeBlinkLeft",
    "EyeBlinkRight",
    "EyeLookDownLeft",
    "EyeLookDownRight",
    "EyeLookInLeft",
    "EyeLookInRight",
    "EyeLookOutLeft",
    "EyeLookOutRight",
    "EyeLookUpLeft",
    "EyeLookUpRight",
    "EyeSquintLeft",
    "EyeSquintRight",
    "EyeWideLeft",
    "EyeWideRight",
    "JawForward",
    "JawLeft",
    "JawOpen",
    "JawRight",
    "MouthClose",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthFrownLeft",
    "MouthFrownRight",
    "MouthFunnel",
    "MouthLeft",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthPucker",
    "MouthRight",
    "MouthRollLower",
    "MouthRollUpper",
    "MouthShrugLower",
    "MouthShrugUpper",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
    "NoseSneerLeft",
    "NoseSneerRight",
    "PupilDilateLeft",
    "PupilDilateRight"
]

# print(len(desired_columns))
print(len(blend_shapes))


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


# Load the CSV file
csv_file = r"..\data\participant_01\participant_01_predicted_blendshapes.csv"
df = pd.read_csv(csv_file, index_col=0)  # Read the CSV file with the first column as index and the first row as headers

df = df.iloc[2400:]  # start from the interesting part

data_to_AU = fill_symetrical(df, mapping, blend_shapes)


video_full_data = pd.read_csv(r"..\data\participant_01\participant_01_avatar_blendshapes.csv", index_col=0)

video_full_data = video_full_data.iloc[2400:]  # start from the interesting part

print(video_full_data.columns)
print(blend_shapes)
video_full_data = fill_symetrical(video_full_data, mapping, blend_shapes)


if __name__ == '__main__':

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
        for row1, row2 in zip(data_to_AU, video_full_data):
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
