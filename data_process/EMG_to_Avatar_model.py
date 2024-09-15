import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timezone, timedelta
import joblib
from data_process.classifying_ica_components import filter_signal
# packages for the model
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error,  r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
# from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import ExtraTreesRegressor

from scipy.stats import ttest_rel
import seaborn as sns

from prepare_data_for_model import *


# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the project folder by going up two levels from the script's directory
project_folder = os.path.abspath(os.path.join(script_dir, '..'))

test_eeg = False
ICA_flag = False
EMG_flag = True
save_results = True
segments_length = 35  # length of each segment in seconds
models = ['LR', 'ETR', 'Ridge']


# Function to evaluate models and print results
def evaluate_models(X_train, X_test, Y_train, Y_test, model_name, data_label, cross_val=True, plot_weights=True):
    print(f"\nEvaluating {model_name} for {data_label} data...")

    # # Use LazyPredict to evaluate multiple models
    # reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    # models, predictions = reg.fit(X_train, X_test, Y_train, Y_test)
    # print(models)

    if model_name == 'LR':
        model = LinearRegression()
    elif model_name == 'ETR':
        model = ExtraTreesRegressor()
    elif model_name == 'Ridge':
        model = Ridge()

    if cross_val:
        # Cross-validation setup
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        Y_pred_cv = cross_val_predict(model, X_train, Y_train, cv=kf)

        mse_cv = mean_squared_error(Y_train, Y_pred_cv)
        r2_cv = r2_score(Y_train, Y_pred_cv)
        mae_cv = mean_absolute_error(Y_train, Y_pred_cv)

        print(f'\n{model_name} Cross-Validation Results for {data_label} data:')
        print(f'Mean Squared Error (CV): {mse_cv}')
        print(f'R2 Score (CV): {r2_cv}')
        print(f'Mean Absolute Error (CV): {mae_cv}')

    # Train the model on the full training set
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)

    print(f'\n{model_name} Results for {data_label} data:')
    print(f'Mean Squared Error: {mse}')
    print(f'R2 Score: {r2}')
    print(f'Mean Absolute Error: {mae}\n')

    if plot_weights and model_name == 'LR':
        # Visualize the Weight Matrix
        weights = model.coef_

        # order of the blendshapes to be from the top of the face to the bottom
        ordered_blendshapes = [
            'BrowInnerUp', 'BrowDownRight', 'BrowOuterUpRight', 'EyeLookUpRight', 'EyeBlinkRight',
            'EyeSquintRight', 'EyeWideRight', 'EyeLookDownRight', 'EyeLookOutRight', 'EyeLookInRight',
            'NoseSneerRight', 'CheekSquintRight', 'CheekPuff', 'MouthFunnel', 'MouthPucker', 'MouthRight',
            'MouthPressRight', 'MouthUpperUpRight', 'MouthRollUpper', 'MouthShrugUpper', 'MouthSmileRight',
            'JawForward', 'JawOpen', 'JawRight', 'MouthStretchRight', 'MouthDimpleRight', 'MouthLowerDownRight',
            'MouthRollLower', 'MouthShrugLower', 'MouthFrownRight', 'MouthClose'
        ]

        # Assuming blendshapes is a list of blendshape names in the original order
        blendshapes_dict = {name: i for i, name in enumerate(blendshapes)}
        ordered_indices = [blendshapes_dict[name] for name in ordered_blendshapes]

        # Reorder the rows of the weights matrix
        weights_ordered = weights[ordered_indices, :]

        plt.figure(figsize=(12, 8))
        sns.heatmap(weights_ordered, annot=True, cmap='coolwarm', vmin=-0.5, vmax=0.5, xticklabels=range(1, X_train.shape[1] + 1),
                    yticklabels=ordered_blendshapes)
        plt.xlabel('ICs')
        plt.ylabel('blendshapes')
        plt.title(f'Weight Matrix for {data_label} data')
        plt.show()

    return model, mse, r2, mae, Y_pred


data_path = fr"{project_folder}\data"
for participant_folder in os.listdir(data_path):
    participant_ID = participant_folder
    participant_folder_path = fr'{data_path}\{participant_folder}'
    for session_folder in os.listdir(participant_folder_path):
        session_folder_path = fr'{participant_folder_path}\{session_folder}'
        session_number = session_folder
        ica_after_order = extract_and_order_ica_data(participant_ID, session_folder_path, session_number)
        # print(ica_after_order.shape)

        edf_path = fr"{session_folder_path}\{participant_ID}_{session_number}.edf"
        if test_eeg or EMG_flag:
            emg_file = mne.io.read_raw_edf(edf_path, preload=True)
        else:
            emg_file = mne.io.read_raw_edf(edf_path, preload=False)

        emg_fs = emg_file.info['sfreq']  # sampling frequency of the emg data

        X_full = ica_after_order
        if EMG_flag:
            X_full = emg_file.get_data()
        #     filter the data
        # normalize
        X_full = normalize_ica_data(X_full)
        print("X_full shape: ", X_full.shape)

        # RMS
        X_full_RMS = sliding_window(X_full, method="RMS", fs=emg_fs).T

        annotations_list = ['05_Forehead', '07_Eye_gentle', '09_Eye_tight', '12_Nose', '14_Smile_closed',
                            '16_Smile_open', '19_Lip_pucker', '21_Cheeks', '23_Snarl', '26_Depress_lip']
        events_timings = get_annotations_timings(emg_file, annotations_list)
        if ICA_flag:
            participant_ica_windows = prepare_relevant_data(ica_after_order, emg_file, emg_fs, events_timings=events_timings,
                                                            segments_length=segments_length, norm="ICA", averaging="RMS")
        if EMG_flag:
            emg_data = emg_file.get_data()
            # filter emg
            emg_data = filter_signal(emg_data, emg_fs)
            participant_ica_windows = prepare_relevant_data(emg_data, emg_file, emg_fs, events_timings=events_timings,
                                                            segments_length=segments_length, norm="ICA", averaging="RMS")

        # add EEG
        if test_eeg:
            participant_ica_windows_with_eeg = add_eeg(emg_file, participant_ica_windows, fs=emg_fs, events_timings=events_timings,
                                                        segments_length=segments_length, norm="ICA", averaging="RMS")


        # Load the avatar data skip the first row (the features names)
        avatar_data = pd.read_csv(fr"{session_folder_path}/"
                                  fr"{participant_ID}_{session_number}_interpolated_relevant_only_right.csv",
                                  header=0, index_col=0)

        blendshapes = avatar_data.columns

        participant_avatar_windows_cut = prepare_avatar_relevant_data(participant_ID, avatar_data, emg_file, participant_ica_windows,
                                                                      fs=60, events_timings=events_timings,
                                                                      segments_length=segments_length, norm=None, averaging="MEAN")


        # model

        # Transpose to get the shape (samples, features)
        X_ica = participant_ica_windows.T
        Y = participant_avatar_windows_cut.T

        # # Impute NaNs in Y data
        # imputer = SimpleImputer(strategy='constant', fill_value=0)
        # Y = imputer.fit_transform(Y)

        # Split the data into training and testing sets for both versions of X data
        X_ica_train, X_ica_test, Y_train, Y_test = train_test_split(X_ica, Y, test_size=0.2, random_state=42)
        # get indices of X_ica_test in X_ica
        X_ica_test_indices = np.isin(X_ica, X_ica_test).all(axis=1)
        # get the indices from the true values
        X_ica_test_indices = np.where(X_ica_test_indices == True)[0]
        if test_eeg:
            X_ica_eeg = participant_ica_windows_with_eeg.T  # Shape: (11679, 18)
            X_ica_eeg_train, X_ica_eeg_test, Y_train, Y_test = train_test_split(X_ica_eeg, Y, test_size=0.2,
                                                                            random_state=42)



        # run the models
        for model in models:

            model_path = fr"{session_folder_path}/{participant_ID}_{session_number}_blendshapes_model_{model}.joblib"
            if EMG_flag:
                model_path = fr"{session_folder_path}/{participant_ID}_{session_number}_blendshapes_model_{model}_EMG.joblib"

            if os.path.exists(model_path):  # Check if the model already exists
                print(f"Model {model} already exists. Skipping to the next model...")
                continue  # Skip the current model and continue with the next one

            # Evaluate models for ICA data
            model_ica, mse_ica, r2_ica, mae_ica, Y_pred_ica = evaluate_models(X_ica_train, X_ica_test, Y_train, Y_test,
                                                                              model, f'{participant_ID} {session_number} ICA')

            # Evaluate models for ICA + EEG data
            if test_eeg:
                model_ica_eeg, mse_ica_eeg, r2_ica_eeg, mae_ica_eeg, Y_pred_ica_eeg = evaluate_models(X_ica_eeg_train, X_ica_eeg_test, Y_train,
                                                                                   Y_test, model, f'{participant_ID} {session_number} ICA + EEG')

                # Compare evaluation metrics
                print("\nComparison of Evaluation Metrics:")
                print(f"MSE Improvement: {mse_ica - mse_ica_eeg}")
                print(f"R2 Improvement: {r2_ica_eeg - r2_ica}")
                print(f"MAE Improvement: {mae_ica - mae_ica_eeg}")

                # Perform statistical significance test (paired t-test) on the predictions
                t_stat, p_value = ttest_rel(Y_pred_ica.flatten(), Y_pred_ica_eeg.flatten())
                print(f"\nPaired t-test Results:")
                print(f"T-statistic: {t_stat}")
                print(f"P-value: {p_value}")
                if p_value < 0.05:
                    print("The improvement is statistically significant.")
                else:
                    print("The improvement is not statistically significant.")

            if save_results:
                # Save the results
                # predict the whole data
                Y_pred_full = model_ica.predict(X_full_RMS)
                path = fr"{session_folder_path}/{participant_ID}_{session_number}_predicted_blendshapes_{model}.csv"
                if EMG_flag:
                    path = fr"{session_folder_path}/{participant_ID}_{session_number}_predicted_blendshapes_{model}_EMG.csv"
                pd.DataFrame(Y_pred_full, columns=blendshapes).to_csv(path)
                print("Predicted data saved as CSV files.")
                # save the model
                joblib.dump(model_ica, model_path)
                print(f"Model {model} saved as joblib file.")


        # save participant avatar windows data as csv to compare to predicted data
        avatar_sliding_window_method = "MEAN"
        full_avatar_data_windows = sliding_window(avatar_data.to_numpy().T, method=avatar_sliding_window_method, fs=60).T
        print("full avatar windows shape before crop:", full_avatar_data_windows.shape)
        # make it the same size as the predicted values
        full_avatar_data_windows = full_avatar_data_windows[:X_full_RMS.shape[0], :]
        pd.DataFrame(full_avatar_data_windows, columns=blendshapes).to_csv(
            fr"{session_folder_path}\{participant_ID}_{session_number}_avatar_blendshapes_{avatar_sliding_window_method}.csv")
        print("Avatar data saved as CSV file.\n")





