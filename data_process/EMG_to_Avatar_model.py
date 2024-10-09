import mne
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timezone, timedelta
import joblib
from data_process.classifying_ica_components import filter_signal
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error,  r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ttest_rel
import seaborn as sns

from prepare_data_for_model import *

torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the project folder by going up two levels from the script's directory
project_folder = os.path.abspath(os.path.join(script_dir, '..'))

test_eeg = False
ICA_flag = False
EMG_flag = True
save_results = True
build_individual_models = True
train_deep_learning_model = True
tune_autoencoder = False
train_linear_transform = False
train_autoencoder = True
criterion = 'mse'
segments_length = 4  # length of each segment in seconds
models = ['LR', 'ETR', 'Ridge', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor', 'RandomForestRegressor']

# Define a neural network model for the linear transformation
class LinearTransformNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearTransformNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class EnhancedTransformNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(EnhancedTransformNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.output_layer(x)


def train_model(model, X_train, Y_train, X_test, Y_test, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    # Move data to the selected device
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, Y_test)
            test_losses.append(test_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    return train_losses, test_losses


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(SimpleAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()  # or nn.Tanh(), depending on your data range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


# Function to train the autoencoder
# def train_autoencoder(model, X_train, X_test, epochs=100, lr=0.001, batch_size=32):
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
#
#     train_losses = []
#     test_losses = []
#
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0
#         for batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch)
#             loss = criterion(outputs, batch)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#
#         train_loss /= len(train_loader)
#         train_losses.append(train_loss)
#
#         model.eval()
#         with torch.no_grad():
#             test_outputs = model(X_test)
#             test_loss = criterion(test_outputs, X_test)
#             test_losses.append(test_loss.item())
#
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss.item():.4f}')
#
#     return train_losses, test_losses




def pearson_correlation_loss(y_pred, y_true):
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    # Compute means
    mean_pred = torch.mean(y_pred)
    mean_true = torch.mean(y_true)

    # Compute variances
    vx = y_pred - mean_pred
    vy = y_true - mean_true

    # Compute correlation
    numerator = torch.sum(vx * vy)
    denominator = torch.sqrt(torch.sum(vx ** 2) * torch.sum(vy ** 2))

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    correlation = numerator / (denominator + epsilon)

    return 1 - correlation


def train_and_evaluate_best_model(model, X_train, X_test, criterion, best_batch_size, best_lr, epochs=500):
    # Ensure data is on the correct device
    X_train = X_train.to(device)
    X_test = X_test.to(device)

    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)

    if criterion == 'mse':
        criterion = nn.MSELoss()
    elif criterion == 'pearson':
        criterion = pearson_correlation_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

    train_losses = []
    test_losses = []

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, X_test)
            test_losses.append(test_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss.item():.4f}')

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_test_outputs = model(X_test)
        final_test_loss = criterion(final_test_outputs, X_test)

    print(f'Final Test Loss: {final_test_loss.item():.4f}')

    # Plot the train and test loss
    plt.figure(figsize=(10, 5))
    plt.plot(test_losses, label='Test Loss')
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss for Autoencoder')
    plt.legend()
    plt.show()

    return model, final_test_outputs




def train_autoencoder_cv(model, X, criterion, epochs=100, n_splits=5, tune_parameter='batch_size', default_batch_size=32, default_lr=0.001):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_model = None
    best_loss = float('inf')

    # Define hyperparameter ranges
    batch_sizes = [16, 32, 64, 128, 256][:n_splits]
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001][:n_splits]

    if tune_parameter == 'batch_size':
        varied_params = batch_sizes
        constant_param = default_lr
        param_name = 'Batch Size'
    elif tune_parameter == 'learning_rate':
        varied_params = learning_rates
        constant_param = default_batch_size
        param_name = 'Learning Rate'
    else:
        raise ValueError("tune_parameter must be either 'batch_size' or 'learning_rate'")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold + 1} - {param_name}: {varied_params[fold]}")

        X_train, X_val = X[train_idx], X[val_idx]

        # Move data to device
        X_train = torch.FloatTensor(X_train).to(device)
        X_val = torch.FloatTensor(X_val).to(device)

        train_dataset = TensorDataset(X_train)

        if tune_parameter == 'batch_size':
            batch_size = varied_params[fold]
            lr = constant_param
        else:
            batch_size = constant_param
            lr = varied_params[fold]
        # create the data loader with device
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = model.to(device)
        if criterion == 'mse':
            criterion = nn.MSELoss()
        elif criterion == 'pearson':
            criterion = pearson_correlation_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                inputs = batch[0]
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, X_val)

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss.item():.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            best_batch_size = batch_size
            best_lr = lr

    model.load_state_dict(best_model)
    return model, best_batch_size, best_lr


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
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models for blendshape prediction")
    parser.add_argument("--data_path", default=f"C:/Users/YH006_new/fEMG_to_avatar/data", help="Path to data directory")
    parser.add_argument("--test_eeg", action="store_true", default=False, help="Test EEG flag")
    parser.add_argument("--ica_flag", action="store_true", default=True, help="Use ICA flag")
    parser.add_argument("--emg_flag", action="store_true", default=False, help="Use EMG flag")
    parser.add_argument("--save_results", action="store_true", default=True, help="Save results flag")
    parser.add_argument("--build_individual_models", action="store_true", default=True, help="Build individual models flag")
    parser.add_argument("--train_deep_learning_model", action="store_true", default=True, help="Train deep learning model flag")
    parser.add_argument("--tune_autoencoder", action="store_true", default=False, help="Tune autoencoder flag")
    parser.add_argument("--train_linear_transform", action="store_true", default=True, help="Train linear transform flag")
    parser.add_argument("--train_enhanced_linear_transform", action="store_true", default=True, help="Train enhanced linear transform flag")
    parser.add_argument("--train_autoencoder", action="store_true", default=False, help="Train autoencoder flag")
    parser.add_argument("--criterion", default='pearson', choices=['mse', 'pearson'], help="Loss criterion")
    parser.add_argument("--segments_length", type=int, default=4, help="Length of segments in seconds")
    parser.add_argument("--models", nargs='+', default=['LR', 'ETR', 'Ridge', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor', 'RandomForestRegressor'], help="Models to evaluate")

    args = parser.parse_args()

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    performance_results = []

    if args.build_individual_models:
        for participant_folder in os.listdir(args.data_path):
            if participant_folder == 'participant_01':
                continue
            if 'csv' in participant_folder:
                continue
            participant_ID = participant_folder
            participant_folder_path = os.path.join(args.data_path, participant_folder)
            for session_folder in os.listdir(participant_folder_path):
                if session_folder == 'S1':
                    continue
                session_folder_path = os.path.join(participant_folder_path, session_folder)
                session_number = session_folder

                # Run for both ICA and EMG configurations
                for config in ['ICA']:
                    print(f"\nRunning {config} configuration for {participant_ID}, session {session_number}")

                    # Load and prepare data (existing code)
                    ica_after_order = extract_and_order_ica_data(participant_ID, session_folder_path, session_number)
                    edf_path = os.path.join(session_folder_path, f"{participant_ID}_{session_number}_edited.edf")
                    emg_file = mne.io.read_raw_edf(edf_path, preload=True)
                    emg_fs = emg_file.info['sfreq']

                    if args.ica_flag:
                        X_full = ica_after_order
                    elif args.emg_flag:
                        X_full = emg_file.get_data()
                        X_full = filter_signal(X_full, emg_fs)

                    X_full = normalize_ica_data(X_full)
                    X_full_RMS = sliding_window(X_full, method="RMS", fs=emg_fs).T

                    # Prepare data for model (existing code)
                    annotations_list = ['05_Forehead', '07_Eye_gentle', '09_Eye_tight', '12_Nose', '14_Smile_closed',
                                        '16_Smile_open', '19_Lip_pucker', '21_Cheeks', '23_Snarl', '26_Depress_lip']
                    # print all the annotations that contains the strings in annotations_list
                    annotations_list_with_start_end = []
                    for annotation in emg_file.annotations.description:
                        for name in annotations_list:
                    #         if annotation contains the name, append the annotation to the list
                            if name in annotation:
                                annotations_list_with_start_end.append(annotation)
                    for annotation in annotations_list_with_start_end:
                        if annotation in annotations_list:
                    #         delete the annotation from the list
                            annotations_list_with_start_end.remove(annotation)
                    events_timings = get_annotations_timings(emg_file, annotations_list_with_start_end)
                    # make events_timings into a list with 10 lists that each contains the start and end of the annotation
                    events_timings = [[events_timings[i], events_timings[i + 1]] for i in range(0, len(events_timings), 2)]

                    if args.ica_flag:
                        relevant_data_train_emg, relevant_data_test_emg = prepare_relevant_data(ica_after_order, emg_file, emg_fs, events_timings,
                                                                                                events_timings=events_timings,
                                                                                                segments_length=args.segments_length, norm="ICA",
                                                                                                averaging="RMS")
                    elif args.emg_flag:
                        relevant_data_train_emg, relevant_data_test_emg = prepare_relevant_data(X_full, emg_file, emg_fs, events_timings,
                                                                                                events_timings=events_timings,
                                                                                                segments_length=args.segments_length, norm="ICA",
                                                                                                averaging="RMS")
                    # Load avatar data (existing code)
                    avatar_data = pd.read_csv(os.path.join(session_folder_path, f"{participant_ID}_{session_number}_interpolated_relevant_only_right.csv"),
                                              header=0, index_col=0)
                    blendshapes = avatar_data.columns
                    relevant_data_train_avatar, relevant_data_test_avatar = prepare_avatar_relevant_data(participant_ID, avatar_data, emg_file,
                                                                                                 relevant_data_train_emg, relevant_data_test_emg,
                                                                                                 events_timings,
                                                                                                 fs=60, events_timings=events_timings,
                                                                                                 segments_length=args.segments_length, norm=None,
                                                                                                 averaging="RMS")
                    # plot ICA components vs avatar blendshapes
                    # plot_ica_vs_blendshapes(avatar_data, blendshapes, emg_file, emg_fs, events_timings, ica_after_order, participant_ID)

                    X_train = relevant_data_train_emg.T
                    X_test = relevant_data_test_emg.T
                    Y_train = relevant_data_train_avatar.T
                    Y_test = relevant_data_test_avatar.T

                    # Standardize the features
                    scaler_X = StandardScaler()
                    scaler_Y = StandardScaler()

                    X_train = scaler_X.fit_transform(X_train)
                    X_test = scaler_X.transform(X_test)
                    Y_train = scaler_Y.fit_transform(Y_train)
                    Y_test = scaler_Y.transform(Y_test)

                    if args.train_deep_learning_model:
                        # Convert to PyTorch tensors and move to the selected device
                        X_train = torch.FloatTensor(X_train).to(device)
                        X_test = torch.FloatTensor(X_test).to(device)
                        Y_train = torch.FloatTensor(Y_train).to(device)
                        Y_test = torch.FloatTensor(Y_test).to(device)

                        # Initialize the model and move it to the selected device
                        input_dim = X_train.shape[1]
                        output_dim = Y_train.shape[1]

                        if args.train_linear_transform:
                            if args.train_enhanced_linear_transform:
                                print("Training Enhanced Linear Transform Model...")
                                model = EnhancedTransformNet(input_dim, output_dim).to(device)
                            else:
                                print("Training Linear Transform Model...")
                                model = LinearTransformNet(input_dim, output_dim).to(device)
                            # Train the model
                            train_losses, test_losses = train_model(model, X_train, Y_train, X_test, Y_test, epochs=30, lr=0.01)

                            # Get the transformation matrix
                            # transformation_matrix = model.linear.weight.detach().cpu().numpy()
                            # Make predictions
                            model.eval()
                            with torch.no_grad():
                                Y_pred = model(X_test)
                            # Move predictions back to CPU for further processing
                            Y_pred = Y_pred.cpu()

                            if args.save_results:
                                model_name = 'LinearTransform'
                                # perform the inverse transformation to get the original values
                                Y_pred = scaler_Y.inverse_transform(Y_pred)
                                pred_path = os.path.join(session_folder_path, f"{participant_ID}_{session_number}_predicted_blendshapes_{model_name}_{config}.csv")
                                pd.DataFrame(Y_pred, columns=blendshapes).to_csv(pred_path)
                                print(f"Predicted data saved as {pred_path}")
                                model_path = os.path.join(session_folder_path, f"{participant_ID}_{session_number}_blendshapes_model_{model_name}_{config}.joblib")
                                # Save model
                                joblib.dump(model, model_path)
                                print(f"Model {model_name} for {config} saved as {model_path}")

                        if args.train_autoencoder:
                            print("Training Autoencoder Model...")
                            encoding_dim = output_dim  # You can adjust this

                            autoencoder = SimpleAutoencoder(input_dim, encoding_dim)

                            if args.tune_autoencoder:
                                # To tune batch size
                                _, best_batch_size, _ = train_autoencoder_cv(autoencoder, X_train.cpu().numpy(), args.criterion, tune_parameter='batch_size')

                                # To tune learning rate
                                _, _, best_lr = train_autoencoder_cv(autoencoder, X_train.cpu().numpy(), args.criterion, tune_parameter='learning_rate')
                            else:
                                best_batch_size = 16
                                best_lr = 0.01
                            print(f"Best batch size: {best_batch_size}, Best learning rate: {best_lr}")
                            # Train the best model
                            autoencoder, Y_pred = train_and_evaluate_best_model(autoencoder, X_train, X_test, args.criterion, best_batch_size, best_lr)

                            # make predictions using the autoencoder
                            autoencoder.eval()
                            with torch.no_grad():
                                Y_pred = autoencoder.encoder(X_test)

                            # Move predictions back to CPU for further processing
                            Y_pred = Y_pred.cpu().numpy()

                            # Save the autoencoder model
                            if args.save_results:
                                model_name = 'Autoencoder'
                                # perform the inverse transformation to get the original values
                                Y_pred = scaler_Y.inverse_transform(Y_pred)
                                pred_path = os.path.join(session_folder_path, f"{participant_ID}_{session_number}_predicted_blendshapes_{model_name}_{config}.csv")
                                pd.DataFrame(Y_pred, columns=blendshapes).to_csv(pred_path)
                                model_path = os.path.join(session_folder_path, f"{participant_ID}_{session_number}_blendshapes_model_{model_name}_{config}.joblib")
                                joblib.dump(autoencoder, model_path)
                                print(f"Model {model_name} for {config} saved as {model_path}")

                    else:
                        # Run models
                        for model_name in args.models:
                            model_path = os.path.join(session_folder_path, f"{participant_ID}_{session_number}_blendshapes_model_{model_name}_{config}.joblib")

                            if args.emg_flag:
                                model, mse, r2, mae, Y_pred = evaluate_models(X_train, X_test, Y_train, Y_test,
                                                                              model_name, f'{participant_ID} {session_number} {config}',
                                                                              is_ICA=False)
                            else:
                                model, mse, r2, mae, Y_pred = evaluate_models(X_train, X_test, Y_train, Y_test,
                                                                              model_name, f'{participant_ID} {session_number} {config}',
                                                                              is_ICA=True)

                            if args.save_results:
                                # perform the inverse transformation to get the original values
                                Y_pred = scaler_Y.inverse_transform(Y_pred)
                                pred_path = os.path.join(session_folder_path, f"{participant_ID}_{session_number}_predicted_blendshapes_{model_name}_{config}.csv")
                                pd.DataFrame(Y_pred, columns=blendshapes).to_csv(pred_path)
                                print(f"Predicted data saved as {pred_path}")

                                # Save model
                                joblib.dump(model, model_path)
                                print(f"Model {model_name} for {config} saved as {model_path}")

                        # After running both configurations, save performance results
                        performance_df = pd.DataFrame(performance_results)
                        performance_csv_path = os.path.join(session_folder_path, f"{participant_ID}_{session_number}_model_performance_comparison.csv")
                        performance_df.to_csv(performance_csv_path, index=False)
                        print(f"Model performance comparison results saved to {performance_csv_path}")

                        # Reset performance_results for the next session
                        performance_results = []

                    plot_predictions_vs_avatar(Y_pred, scaler_Y.inverse_transform(Y_test.cpu()), blendshapes)
                    # convert the test values to the original scale
                    Y_test = scaler_Y.inverse_transform(Y_test.cpu())
                    # Save avatar data (existing code)
                    avatar_sliding_window_method = "RMS"
                    pd.DataFrame(Y_test, columns=blendshapes).to_csv(
                        os.path.join(session_folder_path, f"{participant_ID}_{session_number}_avatar_blendshapes_{avatar_sliding_window_method}.csv"))
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
                edf_path = fr"{session_folder_path}\{participant_ID}_{session_number}_edited.edf"
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


def plot_predictions_vs_avatar(Y_pred, Y_test, blendshapes):
    fig, axs = plt.subplots(6, 6, figsize=(10, 10), dpi=300)
    plt.rcParams.update({'font.size': 12})  # Adjust base font size

    # Calculate the overall time range for both EMG and avatar data
    max_time_emg = max(len(data) for data in Y_pred)
    max_time_avatar = max(len(data) for data in Y_test)
    max_time = max(max_time_emg, max_time_avatar)

    # Find global min and max for both predicted and ground truth
    data_min = min(min(np.min(Y_pred[i]), np.min(Y_test[i])) for i in range(Y_test.shape[1]))
    data_max = max(max(np.max(Y_pred[i]), np.max(Y_test[i])) for i in range(Y_test.shape[1]))

    for idx, (blendshape, original_index) in enumerate(zip(right_blendshapes, range(Y_test.shape[1]))):
        i, j = divmod(idx, 6)
        ax = axs[i, j]

        time_axis_pred = np.arange(len(Y_pred[original_index]))
        time_axis_test = np.arange(len(Y_test[original_index]))

        ax.plot(time_axis_pred, Y_pred[original_index], label='Predicted', color='blue')
        ax.plot(time_axis_test, Y_test[original_index], label='Ground Truth', color='red')

        ax.set_title(blendshape, fontsize=10)
        ax.set_ylim(data_min, data_max)
        ax.set_xlim(0, max_time)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # Adjust tick parameters
        ax.tick_params(axis='both', which='both', labelsize=8)

        # Only show x-axis labels for bottom row
        if i != 5:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=10)

        # Only show y-axis labels for left column
        if j != 0:
            ax.set_yticklabels([])

    # Add legend to the 32nd subplot (lower right corner)
    legend_i, legend_j = 5, 5  # Coordinates for the 32nd subplot
    axs[legend_i, legend_j].legend(fontsize=8, loc='center')
    axs[legend_i, legend_j].set_axis_off()  # Turn off axis for the legend subplot

    # Remove any unused subplots
    for idx in range(len(right_blendshapes), 36):
        i, j = divmod(idx, 6)
        if (i, j) != (legend_i, legend_j):  # Keep the legend subplot
            fig.delaxes(axs[i, j])

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_ica_vs_blendshapes(avatar_data, blendshapes, emg_file, emg_fs, events_timings, ica_after_order, participant_ID):
    emg_fs = int(emg_fs)
    relevant_data_train = []
    for i in range(len(events_timings)):
        relevant_data_train.append(
            ica_after_order[:, int(events_timings[i][0]) * emg_fs:int(events_timings[i][1]) * emg_fs])
    relevant_data_train_emg = np.concatenate(relevant_data_train, axis=1)
    time_delta = get_time_delta(emg_file, avatar_data, participant_ID)
    avatar_data = avatar_data.to_numpy().T
    fs_emg = emg_file.info['sfreq']
    print("original avatar data shape: ", avatar_data.shape)
    avatar_fps = 60  # data collection was in 60 fps
    frames_to_cut = int(time_delta * avatar_fps)
    avatar_data = avatar_data[:, frames_to_cut:]
    print("avatar data cut shape: ", avatar_data.shape)
    avatar_fs = 60
    relevant_data_avatar_train = []
    for i in range(len(events_timings)):
        relevant_data_avatar_train.append(
            avatar_data[:, int(events_timings[i][0]) * avatar_fs:int(events_timings[i][1]) * avatar_fs])
    relevant_data_train_avatar = np.concatenate(relevant_data_avatar_train, axis=1)
    # Define blendshapes to exclude
    # Define blendshapes to exclude
    # Define blendshapes to exclude
    exclude_blendshapes = {'EyeLookInRight', 'NoseSneerRight', 'EyeLookUpRight', 'MouthDimpleRight'}

    # Filter blendshapes to include only those with 'Right' and not in the exclude list
    right_blendshapes = [bs for bs in blendshapes if
                         'Right' in bs and bs not in exclude_blendshapes]
    right_indices = [i for i, bs in enumerate(blendshapes) if
                     'Right' in bs and bs not in exclude_blendshapes]

    fig, axs = plt.subplots(len(right_blendshapes), 2, figsize=(24, 24), dpi=300)
    plt.rcParams.update({'font.size': 28})  # Increase base font size

    # Normalize ICA and blendshape data
    def normalize_data(data):
        return data / np.max(data)

    normalized_emg = [normalize_data(data) for data in relevant_data_train_emg]
    normalized_avatar = [normalize_data(data) for data in relevant_data_train_avatar]

    # Calculate the overall time range for both EMG and avatar data
    max_time_emg = max(len(data) for data in normalized_emg) / emg_fs
    max_time_avatar = max(len(data) for data in normalized_avatar) / avatar_fs
    max_time = max(max_time_emg, max_time_avatar)

    for i, (blendshape, original_index) in enumerate(zip(right_blendshapes, right_indices)):
        # Reverse the order of ICA plots
        ica_index = len(right_blendshapes) - i - 1

        if ica_index < len(normalized_emg):
            time_axis = np.arange(len(normalized_emg[ica_index])) / emg_fs
            axs[i, 0].plot(time_axis, normalized_emg[ica_index], linewidth=0.8)
            axs[i, 0].set_ylabel(f'ICA {ica_index + 1}', rotation=0, ha='right', va='center')

        time_axis = np.arange(len(normalized_avatar[original_index])) / avatar_fs
        axs[i, 1].plot(time_axis, normalized_avatar[original_index], linewidth=0.8)
        axs[i, 1].set_ylabel(blendshape, rotation=0, ha='right', va='center')

        # Set the same x-axis limits for both subplots
        axs[i, 0].set_xlim(0, max_time)
        axs[i, 1].set_xlim(0, max_time)

        # Remove top and right spines, keep only y-axis
        for ax in [axs[i, 0], axs[i, 1]]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labelsize=20)
            ax.tick_params(axis='y', which='both', labelsize=20)  # Make yticks smaller

        # Add horizontal line at y=0 for all subplots
        axs[i, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        axs[i, 1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # Remove x-axis labels for all but the bottom subplot
        if i < len(right_blendshapes) - 1:
            axs[i, 0].set_xticklabels([])
            axs[i, 1].set_xticklabels([])
        else:
            # Add x-axis only for the bottom subplots
            for ax in [axs[i, 0], axs[i, 1]]:
                ax.spines['bottom'].set_visible(True)
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
            axs[i, 0].set_xlabel('Time (s)', size=28)
            axs[i, 1].set_xlabel('Time (s)', size=28)

    # Add ylabels for the 8 subplots
    axs[0, 0].set_title('ICA Components', size=28)
    axs[0, 1].set_title('Blendshapes', size=28)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect parameter to make room for suptitle
    plt.savefig(fr"{project_folder}\results\{participant_ID}_ICA_vs_blendshapes.png")
    plt.close()

if __name__ == "__main__":
    main()
