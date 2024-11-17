import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import joblib
import os

from data_process.prepare_data_for_model import project_folder
from prepare_data_for_model import extract_and_order_ica_data, get_annotations_timings, prepare_relevant_data_new, \
    prepare_avatar_relevant_data
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from EMG_to_Avatar_model import split_concatenated_array, plot_prediction_vs_GT



# Define the model class first
class ImprovedEnhancedTransformNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout_rate=0.4):
        super(ImprovedEnhancedTransformNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)

        self.layer3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)

        self.layer4 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim * 2)

        self.layer5 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.activation(self.bn1(self.layer1(x))))
        x = self.dropout(self.activation(self.bn2(self.layer2(x))))
        x = self.dropout(self.activation(self.bn3(self.layer3(x))))
        x = self.dropout(self.activation(self.bn4(self.layer4(x))))
        x = self.dropout(self.activation(self.bn5(self.layer5(x))))
        return self.output_layer(x)


# Project folder path
project_folder = r"C:\Users\Hila\OneDrive\מסמכים\fEMG_to_avatar"  # Replace with actual path
data_path = fr"{project_folder}\data"  # Replace with actual path

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def plot_correlations_comparison(participant1_ID, participant2_ID, session_number, Y_pred, Y_test, correlations,
                                 project_folder):
    """
    Create a horizontal barplot comparing correlations between predicted and ground truth values
    for each action unit for two participants, displayed one below the other.
    """
    # Calculate correlations for both participants
    correlations1 = []
    correlations2 = correlations
    for i in range(Y_test.shape[1]):
        corr1, _ = pearsonr(Y_pred[:, i], Y_test[:, i])
        correlations1.append(corr1)

    # Save correlation values
    np.save(f"{project_folder}/results/{participant1_ID}_{session_number}_correlations.npy", correlations1)

    # Create figure
    plt.figure(figsize=(6, 10), dpi=300)  # Increased height to accommodate separated bars
    plt.rcParams.update({'font.size': 8})

    # Create horizontal barplot with separated bars
    num_units1 = len(correlations1)
    num_units2 = len(correlations2)

    y_pos1 = np.arange(num_units1 * 2)  # Double the positions for separate bars
    y_pos2 = np.arange(num_units2 * 2)  # Double the positions for separate bars

    # Create bars for both participants
    # Place participant 1's bars at even positions
    bars1 = plt.barh(y_pos1[::2], correlations1, height=0.6,
                     label=f'Participant {participant1_ID}', color='#4287f5')  # Blue color
    # Place participant 2's bars at odd positions
    bars2 = plt.barh(y_pos2[1::2], correlations2, height=0.6,
                     label=f'Participant {participant2_ID}', color='#f54242')  # Red color

    # Add legend
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=2)

    # Customize plot
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Action Unit', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)

    # Set axis limits
    min_corr = min(min(correlations1), min(correlations2))
    max_corr = max(max(correlations1), max(correlations2), 1)
    plt.xlim(min_corr, max_corr)

    # Set y-axis ticks and limits
    # Create labels that show AU number for each pair of bars
    tick_positions = y_pos1[::2] + 0.5  # Center between each pair of bars
    tick_labels = range(1, num_units1 + 1)
    plt.yticks(tick_positions, tick_labels, fontsize=8)
    plt.ylim(-0.5, len(y_pos1) - 0.5)

    # Move x-axis to top
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Remove spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add thin horizontal lines between AU groups
    for i in range(1, num_units1):
        plt.axhline(y=i * 2 - 0.5, color='gray', linestyle='-', linewidth=0.1, alpha=0.3)

    # Adjust layout
    plt.subplots_adjust(top=0.8, left=0.15, right=0.95)

    # Save plot
    fig_path = fr"{project_folder}\results\correlations_comparison_barplot.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()


def test_cross_participant():
    # Load source model (participant_03)
    source_participant = "participant_03"
    source_session = "S1"
    model_name = "ImprovedEnhancedTransformNet"
    source_path = os.path.join(data_path, source_participant, source_session)
    model_path = os.path.join(source_path, f"{source_participant}_{source_session}_blendshapes_{model_name}_trial_1_ICA.joblib")

    # Create model instance with same architecture
    input_dim = 16  # Number of ICA components
    output_dim = 31  # Number of blendshapes
    model = ImprovedEnhancedTransformNet(input_dim, output_dim).to(device)

    # Load saved state dict
    model = joblib.load(model_path)
    model = model.to(device)

    # Load target data (participant_04)
    target_participant = "participant_04"
    target_session = "S1"
    target_path = os.path.join(data_path, target_participant, target_session)

    # Load ICA data
    ica_data = extract_and_order_ica_data(target_participant, target_path, target_session)

    # Load EEG data
    edf_path = os.path.join(target_path, f"{target_participant}_{target_session}_edited.edf")
    emg_file = mne.io.read_raw_edf(edf_path, preload=True)
    emg_fs = emg_file.info['sfreq']

    # Define annotations
    annotations_list = ['05_Forehead', '07_Eye_gentle', '09_Eye_tight', '12_Nose', '14_Smile_closed',
                        '16_Smile_open', '19_Lip_pucker', '21_Cheeks', '23_Snarl', '26_Depress_lip']

    # Get annotations with start/end times
    annotations_list_with_start_end = []
    for annotation in emg_file.annotations.description:
        if ('trial_1' in annotation) and ('start' in annotation or 'end' in annotation):
            if not ('Break' in annotation or 'Face_at_rest' in annotation):
                annotations_list_with_start_end.append(annotation)

    events_timings = get_annotations_timings(emg_file, annotations_list_with_start_end)
    events_timings = [[events_timings[i], events_timings[i + 1]] for i in range(0, len(events_timings), 2)]

    # Prepare ICA data
    relevant_data_train_emg, relevant_data_test_emg, rand_lst, test_data_timing = prepare_relevant_data_new(
        ica_data, emg_fs, events_timings, rand_test=True, num_repetition=1, for_plot_flag=False, averaging="RMS")

    # Load and prepare avatar data
    avatar_data = pd.read_csv(os.path.join(target_path,
                                           f"{target_participant}_{target_session}_interpolated_relevant_only_right.csv"),
                              header=0, index_col=0)
    blendshapes = avatar_data.columns

    relevant_data_train_avatar, relevant_data_test_avatar = prepare_avatar_relevant_data(
        target_participant, avatar_data, emg_file, events_timings, rand_test=True,
        num_repetition=1, for_plot_flag=False, rand_lst=rand_lst, fs=60, averaging="RMS")

    # Prepare test data
    X_test = np.concatenate(relevant_data_test_emg, axis=1)
    Y_test = np.concatenate(relevant_data_test_avatar, axis=1)
    Y_test = Y_test[:, :X_test.shape[1]]

    X_test = X_test.T
    Y_test = Y_test.T

    # Scale data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_test = scaler_X.fit_transform(X_test)
    Y_test = scaler_Y.fit_transform(Y_test)

    # Convert to PyTorch tensors
    X_test = torch.FloatTensor(X_test).to(device)
    Y_test = torch.FloatTensor(Y_test).to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test)

    # Process predictions
    Y_pred = Y_pred.cpu().numpy()
    Y_pred = scaler_Y.inverse_transform(Y_pred)
    Y_test = scaler_Y.inverse_transform(Y_test.cpu().numpy())


    # load the correlation values from participant_03
    correlations = np.load(os.path.join(project_folder, "results", f"{source_participant}_{source_session}_correlations.npy"))

    plot_correlations_comparison(source_participant, target_participant, target_session,
                                  Y_pred, Y_test, correlations, project_folder)

    # Save predictions
    pd.DataFrame(Y_pred, columns=blendshapes).to_csv(
        os.path.join(target_path, f"{target_participant}_{target_session}_cross_participant_predictions.csv"))

    # Split predictions and ground truth back to original lengths
    original_lengths = [arr.shape[1] for arr in relevant_data_test_avatar]
    predictions_list = split_concatenated_array(Y_pred.T, original_lengths)
    ground_truth_list = split_concatenated_array(Y_test.T, original_lengths)

    # Inverse transform the scaled data
    Y_pred = scaler_Y.inverse_transform(Y_pred)
    Y_test = scaler_Y.inverse_transform(Y_test)

    # Plot predictions vs ground truth
    plot_prediction_vs_GT(annotations_list, emg_fs, target_participant,
                          ground_truth_list, predictions_list, target_session,
                          test_data_timing, rand_test=True, num_repetition=1)

    return correlations, Y_pred, Y_test


if __name__ == "__main__":
    correlations, predictions, ground_truth = test_cross_participant()
    print(f"Average correlation across blendshapes: {np.mean(correlations):.3f}")