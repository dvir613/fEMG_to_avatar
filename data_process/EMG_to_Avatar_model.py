import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timezone, timedelta
import joblib
from data_process.classifying_ica_components import filter_signal
# packages for the model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error,  r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
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
build_individual_models = True
segments_length = 4  # length of each segment in seconds
models = ['LR', 'ETR', 'Ridge', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor', 'RandomForestRegressor']

performance_results = []
# Function to evaluate models and print results
def evaluate_models(X_train, X_test, Y_train, Y_test, model_name, data_label, cross_val=True, plot_weights=True, is_ICA=True):
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
    elif model_name == 'Lasso':
        model = Lasso()
    elif model_name == 'ElasticNet':
        model = ElasticNet()
    elif model_name == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor()
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor()

    if cross_val:
        # Cross-validation setup
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        if np.isnan(Y_train).any():
            print("Warning: NaN values found in the target variable. Imputing with most frequent value.")
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

    #     save the weights matrix
        if os.path.exists(fr"{project_folder}\results") is False:
            os.makedirs(fr"{project_folder}\results")
        weights_path = fr"{project_folder}\results\{participant_folder}_{session_folder}_{model_name}_weights_{data_label}.npy"
        np.save(weights_path, weights)

    data_name = 'ICA' if is_ICA else 'EMG'
    performance = {
        f'{data_name}_Model': model_name,
        f'{data_name}_Data': data_label,
        f'{data_name}_MSE': mse,
        f'{data_name}_R2': r2,
        f'{data_name}_MAE': mae
    }
    if cross_val:
        performance.update({
            f'{data_name}_MSE_CV': mse_cv,
            f'{data_name}_R2_CV': r2_cv,
            f'{data_name}_MAE_CV': mae_cv
        })
    performance_results.append(performance)

    return model, mse, r2, mae, Y_pred

if build_individual_models:
    data_path = fr"{project_folder}\data"
    for participant_folder in os.listdir(data_path):
        if 'csv' in participant_folder:
            continue
        participant_ID = participant_folder
        participant_folder_path = fr'{data_path}\{participant_folder}'
        for session_folder in os.listdir(participant_folder_path):
            session_folder_path = fr'{participant_folder_path}\{session_folder}'
            session_number = session_folder

            # Run for both ICA and EMG configurations
            for config in ['ICA', 'EMG']:
                if config == 'ICA':
                    ICA_flag = True
                    EMG_flag = False
                else:
                    ICA_flag = False
                    EMG_flag = True

                print(f"\nRunning {config} configuration for {participant_ID}, session {session_number}")

                # Load and prepare data (existing code)
                ica_after_order = extract_and_order_ica_data(participant_ID, session_folder_path, session_number)
                edf_path = fr"{session_folder_path}\{participant_ID}_{session_number}.edf"
                emg_file = mne.io.read_raw_edf(edf_path, preload=True)
                emg_fs = emg_file.info['sfreq']

                if ICA_flag:
                    X_full = ica_after_order
                elif EMG_flag:
                    X_full = emg_file.get_data()
                    X_full = filter_signal(X_full, emg_fs)

                X_full = normalize_ica_data(X_full)
                X_full_RMS = sliding_window(X_full, method="RMS", fs=emg_fs).T

                # Prepare data for model (existing code)
                annotations_list = ['05_Forehead', '07_Eye_gentle', '09_Eye_tight', '12_Nose', '14_Smile_closed',
                                    '16_Smile_open', '19_Lip_pucker', '21_Cheeks', '23_Snarl', '26_Depress_lip']
                events_timings = get_annotations_timings(emg_file, annotations_list)
                trials_lst = [[2,8,16], [3,8,17], [4,10,18], [2,8,14], [2,9,17], [2,6,13], [2,8,14], [2,9,17], [2,6,11], [2,9,15]]
                trials_lst_timing = [[] for i in range(len(trials_lst))]
                for i in range(len(trials_lst)):
                    for j in range(len(trials_lst[0])):
                        trials_lst_timing[i].append(round(events_timings[i] + trials_lst[i][j]))

                if ICA_flag:
                    relevant_data_train_emg, relevant_data_test_emg = prepare_relevant_data(ica_after_order, emg_file, emg_fs, trials_lst_timing,
                                                                                            events_timings=events_timings,
                                                                                            segments_length=4, norm="ICA",
                                                                                            averaging="RMS")
                elif EMG_flag:
                    relevant_data_train_emg, relevant_data_test_emg = prepare_relevant_data(X_full, emg_file, emg_fs, trials_lst_timing,
                                                                                            events_timings=events_timings,
                                                                                            segments_length=4, norm="ICA",
                                                                                            averaging="RMS")
                # Load avatar data (existing code)
                avatar_data = pd.read_csv(fr"{session_folder_path}/"
                                          fr"{participant_ID}_{session_number}_interpolated_relevant_only_right.csv",
                                          header=0, index_col=0)
                blendshapes = avatar_data.columns
                relevant_data_train_avatar, relevant_data_test_avatar = prepare_avatar_relevant_data(participant_ID, avatar_data, emg_file,
                                                                                             relevant_data_train_emg, relevant_data_test_emg,
                                                                                             trials_lst_timing,
                                                                                             fs=60, events_timings=events_timings,
                                                                                             segments_length=4, norm=None,
                                                                                             averaging="MEAN")
                X_train = relevant_data_train_emg.T
                X_test = relevant_data_test_emg.T
                Y_train = relevant_data_train_avatar.T
                Y_test = relevant_data_test_avatar.T
                # Run models
                for model_name in models:
                    model_path = fr"{session_folder_path}/{participant_ID}_{session_number}_blendshapes_model_{model_name}_{config}.joblib"

                    if EMG_flag:
                        model, mse, r2, mae, Y_pred = evaluate_models(X_train, X_test, Y_train, Y_test,
                                                                      model_name, f'{participant_ID} {session_number} {config}',
                                                                      is_ICA=False)
                    else:
                        model, mse, r2, mae, Y_pred = evaluate_models(X_train, X_test, Y_train, Y_test,
                                                                      model_name, f'{participant_ID} {session_number} {config}',
                                                                      is_ICA=True)

                    if save_results:
                        pred_path = fr"{session_folder_path}/{participant_ID}_{session_number}_predicted_blendshapes_{model_name}_{config}.csv"
                        pd.DataFrame(Y_pred, columns=blendshapes).to_csv(pred_path)
                        print(f"Predicted data saved as {pred_path}")

                        # Save model
                        joblib.dump(model, model_path)
                        print(f"Model {model_name} for {config} saved as {model_path}")

            # After running both configurations, save performance results
            performance_df = pd.DataFrame(performance_results)
            performance_csv_path = fr"{session_folder_path}/{participant_ID}_{session_number}_model_performance_comparison.csv"
            performance_df.to_csv(performance_csv_path, index=False)
            print(f"Model performance comparison results saved to {performance_csv_path}")

            # Reset performance_results for the next session
            performance_results = []

            # Save avatar data (existing code)
            avatar_sliding_window_method = "MEAN"
            pd.DataFrame(Y_test, columns=blendshapes).to_csv(
                fr"{session_folder_path}\{participant_ID}_{session_number}_avatar_blendshapes_{avatar_sliding_window_method}.csv")
            print("Avatar data saved as CSV file.\n")
else:

    data_path = fr"{project_folder}\data"

    # Initialize empty lists to store combined data
    X_train_combined = []
    X_test_combined = []
    Y_train_combined = []
    Y_test_combined = []

    # Loop through all participants and sessions to collect data
    for participant_folder in os.listdir(data_path):
        participant_ID = participant_folder
        participant_folder_path = fr'{data_path}\{participant_folder}'
        for session_folder in os.listdir(participant_folder_path):
            session_folder_path = fr'{participant_folder_path}\{session_folder}'
            session_number = session_folder

            print(f"\nProcessing data for {participant_ID}, session {session_number}")

            # Load and prepare data (existing code)
            ica_after_order = extract_and_order_ica_data(participant_ID, session_folder_path, session_number)
            edf_path = fr"{session_folder_path}\{participant_ID}_{session_number}.edf"
            emg_file = mne.io.read_raw_edf(edf_path, preload=True)
            emg_fs = emg_file.info['sfreq']

            X_full = ica_after_order
            X_full = normalize_ica_data(X_full)
            X_full_RMS = sliding_window(X_full, method="RMS", fs=emg_fs).T

            # Prepare data for model (existing code)
            annotations_list = ['05_Forehead', '07_Eye_gentle', '09_Eye_tight', '12_Nose', '14_Smile_closed',
                                '16_Smile_open', '19_Lip_pucker', '21_Cheeks', '23_Snarl', '26_Depress_lip']
            events_timings = get_annotations_timings(emg_file, annotations_list)
            trials_lst = [[2, 8, 16], [3, 8, 17], [4, 10, 18], [2, 8, 14], [2, 9, 17], [2, 6, 13], [2, 8, 14],
                          [2, 9, 17], [2, 6, 11], [2, 9, 15]]
            trials_lst_timing = [[] for i in range(len(trials_lst))]
            for i in range(len(trials_lst)):
                for j in range(len(trials_lst[0])):
                    trials_lst_timing[i].append(round(events_timings[i] + trials_lst[i][j]))

            relevant_data_train_emg, relevant_data_test_emg = prepare_relevant_data(ica_after_order, emg_file, emg_fs,
                                                                                    trials_lst_timing,
                                                                                    events_timings=events_timings,
                                                                                    segments_length=4, norm="ICA",
                                                                                    averaging="RMS")

            # Load avatar data (existing code)
            avatar_data = pd.read_csv(fr"{session_folder_path}/"
                                      fr"{participant_ID}_{session_number}_interpolated_relevant_only_right.csv",
                                      header=0, index_col=0)
            blendshapes = avatar_data.columns
            relevant_data_train_avatar, relevant_data_test_avatar = prepare_avatar_relevant_data(participant_ID,
                                                                                                 avatar_data, emg_file,
                                                                                                 relevant_data_train_emg,
                                                                                                 relevant_data_test_emg,
                                                                                                 trials_lst_timing,
                                                                                                 fs=60,
                                                                                                 events_timings=events_timings,
                                                                                                 segments_length=4,
                                                                                                 norm=None,
                                                                                                 averaging="MEAN")

            # Append data to combined lists
            X_train_combined.append(relevant_data_train_emg.T)
            X_test_combined.append(relevant_data_test_emg.T)
            Y_train_combined.append(relevant_data_train_avatar.T)
            Y_test_combined.append(relevant_data_test_avatar.T)

    # Combine data from all participants
    X_train = np.vstack(X_train_combined)
    X_test = np.vstack(X_test_combined)
    Y_train = np.vstack(Y_train_combined)
    Y_test = np.vstack(Y_test_combined)

    print(f"Combined training data shape: {X_train.shape}")
    print(f"Combined testing data shape: {X_test.shape}")

    # Run models on combined data
    for model_name in models:
        print(f"\nTraining and evaluating {model_name} on combined participant data")
        model, mse, r2, mae, Y_pred = evaluate_models(X_train, X_test, Y_train, Y_test,
                                                      model_name, "Combined Participants ICA",
                                                      is_ICA=True)

        # Save combined model
        if os.path.exists(fr"{project_folder}\models") is False:
            os.makedirs(fr"{project_folder}\models")
        model_path = fr"{project_folder}\models\combined_{model_name}_ICA.joblib"
        joblib.dump(model, model_path)
        print(f"Combined model {model_name} for ICA saved as {model_path}")

        # Save predictions
        if os.path.exists(fr"{project_folder}\predictions") is False:
            os.makedirs(fr"{project_folder}\predictions")
        pred_path = fr"{project_folder}\predictions\combined_predicted_blendshapes_{model_name}_ICA.csv"
        pd.DataFrame(Y_pred, columns=blendshapes).to_csv(pred_path)
        print(f"Combined predictions saved as {pred_path}")

    # Save performance results
    performance_df = pd.DataFrame(performance_results)
    if os.path.exists(fr"{project_folder}\results") is False:
        os.makedirs(fr"{project_folder}\results")
    performance_csv_path = fr"{project_folder}\results\combined_model_performance_comparison.csv"
    performance_df.to_csv(performance_csv_path, index=False)
    print(f"Combined model performance comparison results saved to {performance_csv_path}")

    # Save combined test data
    pd.DataFrame(Y_test, columns=blendshapes).to_csv(
        fr"{project_folder}\data\combined_avatar_blendshapes_MEAN.csv")
    print("Combined avatar test data saved as CSV file.")


