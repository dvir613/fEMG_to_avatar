import csv
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

folder_name = 'Hila'  # the name of the folder of the participant's data
participant_number = '02'  # the number of the participant (next times it should be identical to the folder name
scene = '001'  # the scene number of the recording
session = '2'  # the session of the recording

anim_file_path = fr"Assets\{folder_name}\{folder_name}\SampleHead [{folder_name}] [00{session}].anim"
asset_file_path = fr"Assets\{folder_name}\[{scene}] {folder_name} [00{session}].asset"
output_csv_path = fr'..\data\participant_{participant_number}\S{session}\participant_{participant_number}_S{session}_interpolated_relevant_only_right.csv'

# # List of relevant blendshapes (right side of the face and brows
# relevant_blendshapes = ['BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff',
#                         'CheekSquintRight', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight',
#                         'EyeLookUpRight', 'EyeSquintRight', 'EyeWideRight', 'JawForward', 'JawOpen', 'JawRight', 'MouthClose',
#                         'MouthDimpleRight', 'MouthFrownRight', 'MouthFunnel', 'MouthLowerDownRight', 'MouthPressRight',
#                         'MouthPucker', 'MouthRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper',
#                         'MouthSmileRight', 'MouthStretchRight', 'MouthUpperUpRight', 'NoseSneerRight']
# relevant blendshapes withouht any of the left side of the face
relevant_blendshapes = ['BrowDownRight', 'BrowInnerUp', 'BrowOuterUpRight', 'CheekPuff',
                        'CheekSquintRight', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight',
                        'EyeLookUpRight', 'EyeSquintRight', 'EyeWideRight', 'JawForward', 'JawOpen', 'JawRight', 'MouthClose',
                        'MouthDimpleRight', 'MouthFrownRight', 'MouthFunnel', 'MouthLowerDownRight', 'MouthPressRight',
                        'MouthPucker', 'MouthRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper',
                        'MouthSmileRight', 'MouthStretchRight', 'MouthUpperUpRight', 'NoseSneerRight']


# function to extract the blendshapes as a dataframe from the anim file
def extract_blendshape_data(anim_file_path, asset_file_path, output_csv_path):
    blendshape_data = {}
    blendshape_data['temp'] = {}

    # Read the .anim file
    with open(anim_file_path, 'r') as file:
        lines = file.readlines()

        current_blendshape = []
        # Parse blendshape data
        for line in range(0, len(lines)):
            if lines[line].startswith('    attribute:'):
                current_blendshape = lines[line].split('_')[-1].strip()
                # If the attribute contains "BlendShapes.", remove it
                if current_blendshape == 'BlendShapesEnabled':  # irrelevant data
                    break  # stop parsing the file
                if 'BlendShapes.' in current_blendshape:
                    current_blendshape = current_blendshape.replace('BlendShapes.', '')
                if current_blendshape not in blendshape_data and current_blendshape in relevant_blendshapes:
                    blendshape_data[current_blendshape] = blendshape_data.pop('temp')  # move the temporary data to the new key
                blendshape_data['temp'] = {}  # start a temporary key to store the blendshape data
            elif lines[line].strip().startswith('time:'):
                time = float(lines[line].split(':')[-1])
                blendshape_data['temp'][time] = {'Value': '', 'Slope': ''}  # start a temporary key to store the blendshape data
                value_line = lines[line + 1].strip().split(':')[-1]
                value = float(value_line.split(':')[-1])
                # sometimes the first inSlope and the last outSlope values are automatically "0", I take both values and check if one of them is not zero.
                inSlope_line = lines[line + 2].strip().split(':')[-1]
                inSlope = float(inSlope_line.split(':')[-1])
                outSlope_line = lines[line + 3].strip().split(':')[-1]
                outSlope = float(outSlope_line.split(':')[-1])
                slope = inSlope if abs(inSlope) > abs(outSlope) else outSlope
                blendshape_data['temp'][time]['Value'] = value
                blendshape_data['temp'][time]['Slope'] = slope

    blendshape_data.pop('temp')  # remove the temporary key

    # Initialize a dictionary to hold the DataFrame data
    data_dict = {'Timestamp': []}
    for blendshape_name in blendshape_data.keys():
        data_dict[f'{blendshape_name}_Value'] = []
        data_dict[f'{blendshape_name}_Slope'] = []

    # Get all timestamps
    timestamps = set()
    for values in blendshape_data.values():
        timestamps.update(values.keys())

    # Sort timestamps
    sorted_timestamps = sorted(timestamps)

    # # Get the maximum timestamp
    # max_timestamp = sorted_timestamps[-1]
    #
    # # Generate all timestamps from 0 to max_timestamp at 60Hz (16.66666667 ms intervals)
    # timestamps = np.arange(0, max_timestamp + 1 / 60, 1 / 60)
    #
    # # Ensure timestamps have 8 decimal places
    # formatted_timestamps = [round(ts, 7) for ts in timestamps]

    # Populate the dictionary with values and slopes for each timestamp
    for timestamp in sorted_timestamps:
        data_dict['Timestamp'].append(timestamp)
        for blendshape_name, values in blendshape_data.items():
            value = values.get(timestamp, {}).get('Value', '')
            slope = values.get(timestamp, {}).get('Slope', '')
            data_dict[f'{blendshape_name}_Value'].append(value)
            data_dict[f'{blendshape_name}_Slope'].append(slope)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data_dict)
    # Convert all columns except 'Timestamp' to numeric types
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # check for missing rows (timestamps)
    # iterate over the first column (timestamps)
    index = 1
    missing_rows = 0
    while index < len(df):
        if round(df.loc[index, 'Timestamp'] - df.loc[index - 1, 'Timestamp'], 5) != round(1/60, 5)\
                and round(df.loc[index, 'Timestamp'] - df.loc[index - 1, 'Timestamp'], 4) != round(1/60, 4)\
                and round(df.loc[index, 'Timestamp'] - df.loc[index - 1, 'Timestamp'], 3) != round(1/60, 3):

            # check if it's a timestamp is miscalculated from the 60Hz sampling rate (16.66666667 ms intervals)
            if not round(df.loc[index, 'Timestamp'] * 60, 5).is_integer():
                print("miscalculated timestamp at", index, df.loc[index, 'Timestamp'])
                # calculate the correct timestamp
                df.loc[index, 'Timestamp'] = round(df.loc[index - 1, 'Timestamp'] + 1 / 60, 5)
                print("corrected timestamp at", index, df.loc[index, 'Timestamp'])
            else:
                print("missing row at", index, df.loc[index, 'Timestamp'])
                missing_value = round(df.loc[index - 1, 'Timestamp'] + 1 / 60, 5)
                if missing_value < df.loc[index, 'Timestamp']:  # if its greater it means the current value is ok
                    # Create a new row with the missing timestamp and NaN values for other columns
                    new_row = pd.DataFrame([[missing_value] + [np.nan] * (len(df.columns) - 1)],
                                       columns=df.columns)
                    missing_rows += 1

                    # Insert the new row by concatenating the original DataFrame with the new row
                    df = pd.concat([df.iloc[:index], new_row, df.iloc[index:]]).reset_index(drop=True)
                    print("new row added at", index, df.loc[index, 'Timestamp'])

        index += 1

    print(missing_rows, "missing rows")
    print("df completed")

    # # fill in the possible missing values
    fill_df(df)

    # Drop all "Slope" columns
    df = df.drop(columns=[col for col in df.columns if 'Slope' in col])

    # Remove the "_Value" suffix from column names
    df = df.rename(columns=lambda x: x.replace('_Value', ''))

    # Replace all NaN values in the first and last rows with 0
    df.iloc[0] = df.iloc[0].fillna(0)
    df.iloc[-1] = df.iloc[-1].fillna(0)

    # Replace all negative values with 0
    df[df < 0] = 0

    # Replace NaN values by linear interpolation column by column
    df = df.interpolate(method='linear', axis=0, limit_direction='both')

    # Read the asset file content
    with open(asset_file_path, 'r') as file:
        asset_content = file.read()
    # extract recording start time
    hours = int(re.search(r'm_Hours:\s+(\d+)', asset_content).group(1))
    minutes = int(re.search(r'm_Minutes:\s+(\d+)', asset_content).group(1))
    seconds = int(re.search(r'm_Seconds:\s+(\d+)', asset_content).group(1))
    frames = int(re.search(r'm_Frames:\s+(\d+)', asset_content).group(1))
    frame_rate_numerator = int(re.search(r'm_Numerator:\s+(\d+)', asset_content).group(1))

    # Calculate milliseconds from frames
    milliseconds = (frames / frame_rate_numerator) * 1000

    # Create the start time using datetime and timedelta
    start_time = datetime(1, 1, 1, hours, minutes, seconds) + timedelta(milliseconds=milliseconds)

    # Format the start time as a string
    start_time_str = start_time.strftime('%H:%M:%S.%f')[:-3]
    df['Timestamp'][0] = start_time_str  # replace the first timestamp with the start time
    print("start time is", start_time_str)

    # Write the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)

    print(f"Conversion to CSV complete. CSV file saved at: {output_csv_path}")


def fill_df(df):
    # calculate the frequency based on the timestamp
    freq = round(1 / df.loc[1, 'Timestamp'], 2)
    print("Frequency is", freq)

    # Replace all zeros with NaNs
    df.replace(0, np.nan, inplace=True)  # it seems that 0 is a default value and is not correct

    start_column_index = 1
    total_columns = len(df.columns)

    filled_values = 0  # count the number of filled values
    empty_values = 0  # count the number of empty values

    for i in range(start_column_index, total_columns, 2):  # iterate through the columns in pairs
        value_column = df.columns[i]
        slope_column = df.columns[i + 1]
        print(value_column)
        # Iterate through the rows of the DataFrame
        index = 0
        empty_values_idx = 0 # follow the indices of the empty values to avoid double counting
        while index < len(df):
            # Check for missing values in the value column
            if pd.isna(df.at[index, value_column]):
                if empty_values_idx == index:
                    empty_values += 1
                # Flag to track whether the condition for index - 2 has been met
                condition_met = False

                # Calculate the corresponding value in the value column using the provided formula
                if index == 0 and len(df) > 2:
                    # If it's the first row, use the next rows values in the right column
                    if not pd.isna(df.at[index + 2, value_column]) and not pd.isna(df.at[index + 1, slope_column]):
                        # if the next slope and the next+1 value exist
                        df.at[index, value_column] = round(
                            df.at[index + 2, value_column] - df.at[index + 1, slope_column] / (freq / 2), 8)
                        filled_values += 1
                    elif not pd.isna(df.at[index + 1, value_column]) and not pd.isna(df.at[index, slope_column]):
                        # if the slope and the next value exist
                        df.at[index, value_column] = round(
                            df.at[index + 1, value_column] - df.at[index, slope_column] / (freq), 8)
                        filled_values += 1

                else:
                    if index > 1:  # so it could calculate based on the value 2 rows before
                        if not pd.isna(df.at[index - 2, value_column]) and not pd.isna(
                                df.at[index - 1, slope_column]):
                            # if the previous slope and the previous+1 value exist
                            df.at[index, value_column] = round(
                                df.at[index - 2, value_column] + df.at[index - 1, slope_column] / (freq / 2), 8)
                            filled_values += 1
                            condition_met = True

                    if not condition_met and index < len(df) - 3:
                        if not pd.isna(df.at[index + 2, value_column]) and not pd.isna(
                                df.at[index + 1, slope_column]):
                            # if the next slope and the next+1 value exist
                            df.at[index, value_column] = round(
                                df.at[index + 2, value_column] - df.at[index + 1, slope_column] / (freq / 2), 8)
                            filled_values += 1
                            if index > 1:
                                index -= 3  # in this case the filled value could be used to fill up the value 2 rows before
                            elif index == 1:
                                index -= 2  # in this case the filled value could be used to fill up the value of the first row

            index += 1
            if empty_values_idx < index:
                empty_values_idx += 1

    print("started with", empty_values, " empty values")
    print(filled_values, "values filled")


# Usage example:
extract_blendshape_data(anim_file_path, asset_file_path, output_csv_path)


