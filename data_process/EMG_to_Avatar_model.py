import mne
import argparse
import json
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ttest_rel, pearsonr
import seaborn as sns
from scipy import signal
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import uniform_filter1d, median_filter

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
    def get_transform_matrix(self):
        return self.output_layer.weight.detach().cpu().numpy()


def train_improved_model(model, X_train, Y_train, X_val, Y_val, criterion, Early_stopping, epochs=100, lr=0.01, patience=10):
    if criterion == 'mse':
        criterion = nn.MSELoss()
    elif criterion == 'pearson':
        criterion = pearson_correlation_loss
    elif criterion == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Added L2 regularization
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Shuffle the training data
        indices = torch.randperm(X_train.size(0))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]

        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_shuffled)
        loss = criterion(outputs, Y_train_shuffled)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, Y_val)
            val_losses.append(val_loss.item())

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # if epochs_no_improve == patience:
        #     print(f'Early stopping triggered at epoch {epoch + 1}')
        #     break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    return model, train_losses, val_losses, val_outputs


def load_best_params(filename='best_params.json'):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                content = f.read().strip()
                if content:  # Check if the file is not empty
                    return json.loads(content)
                else:
                    print(f"Warning: {filename} is empty.")
                    return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filename}: {e}")
            return None
    return None


def save_best_params(best_params, filename='best_params.json'):
    with open(filename, 'w') as f:
        json.dump(best_params, f)


def tune_hyperparameters(X, Y, input_dim, output_dim, criterion, n_splits=5, epochs_list=[50, 100, 200, 300, 400, 500],
                         lr_list=[0.001, 0.01, 0.1], hidden_dim_list=[32, 64, 128], filename='best_params.json',
                         model_name='EnhancedTransformNet'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_params = None
    best_score = float('inf')
    best_losses = []

    for epochs in epochs_list:
        for lr in lr_list:
            for hidden_dim in hidden_dim_list:
                scores = []
                fold_losses = []
                for train_index, val_index in kf.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    Y_train, Y_val = Y[train_index], Y[val_index]
                    if model_name == 'EnhancedTransformNet':
                        model = EnhancedTransformNet(input_dim, output_dim, hidden_dim).to(device)
                    elif model_name == 'LinearTransformNet':
                        model = LinearTransformNet(input_dim, output_dim).to(device)
                    elif model_name == 'ImprovedEnhancedTransformNet':
                        model = ImprovedEnhancedTransformNet(input_dim, output_dim, hidden_dim).to(device)
                    train_losses, val_losses = train_model(model, X_train, Y_train, X_val, Y_val, criterion,
                                                           epochs=epochs, lr=lr)
                    scores.append(val_losses[-1])
                    fold_losses.append((train_losses, val_losses))

                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {'epochs': epochs, 'lr': lr, 'hidden_dim': hidden_dim}
                    best_losses = fold_losses

    print(f"Best parameters: {best_params}")
    print(f"Best validation score: {best_score:.4f}")
    save_best_params(best_params, filename)
    return best_params, best_losses


def plot_loss_values(best_losses, best_params, n_splits=5):
    fig, axes = plt.subplots(n_splits, 1, figsize=(12, 4 * n_splits), sharex=True)
    fig.suptitle(f"Training and Validation Loss for Best Model\nParameters: {best_params}")

    for i, (train_losses, val_losses) in enumerate(best_losses):
        ax = axes[i]
        ax.plot(train_losses, label='Training Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.set_title(f'Fold {i + 1}')
        ax.set_ylabel('Loss')
        ax.legend()

    axes[-1].set_xlabel('Epochs')
    plt.tight_layout()
    plt.show()


def test_best_model(X_train, Y_train, X_test, Y_test, input_dim, output_dim, best_params, criterion, Early_stopping,
                    model_name='EnhancedTransformNet'):
    if model_name == 'EnhancedTransformNet':
        model = EnhancedTransformNet(input_dim, output_dim, best_params['hidden_dim']).to(device)
    elif model_name == 'LinearTransformNet':
        model = LinearTransformNet(input_dim, output_dim).to(device)
    elif model_name == 'ImprovedEnhancedTransformNet':
        model = ImprovedEnhancedTransformNet(input_dim, output_dim, best_params['hidden_dim'], 0.4).to(device)
    model, train_losses, val_losses, val_outputs = train_improved_model(model, X_train, Y_train, X_test, Y_test, criterion,
                                                                        Early_stopping, epochs=best_params['epochs'],
                                                                        lr=best_params['lr'])
    return model, train_losses, val_losses, val_outputs


def plot_model_performance(train_losses, test_losses, dropout_rate=None):
    plt.figure(figsize=(10, 10))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if dropout_rate is not None:
        plt.title(f'Model Performance with Dropout Rate: {dropout_rate}')
    else:
        plt.title('Model Performance')
    plt.legend()
    plt.show()


def train_model(model, X_train, Y_train, X_test, Y_test, criterion, epochs=100, lr=0.01, device='cuda'):
    if criterion == 'mse':
        criterion = nn.MSELoss()
    elif criterion == 'pearson':
        criterion = pearson_correlation_loss
    elif criterion == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    # Move data to the selected device
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    for epoch in range(epochs):
        # Shuffle the training data
        indices = torch.randperm(X_train.size(0))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]

        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_shuffled)
        loss = criterion(outputs, Y_train_shuffled)
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
    vx = torch.subtract(y_pred, mean_pred)
    vy = torch.subtract(y_true, mean_true)

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
    elif criterion == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss()

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


def train_autoencoder_cv(model, X, criterion, epochs=100, n_splits=5, tune_parameter='batch_size',
                         default_batch_size=32, default_lr=0.001):
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
def evaluate_models(X_train, X_test, Y_train, Y_test, model_name, data_label, cross_val=True, plot_weights=True,
                    is_ICA=True):
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
        sns.heatmap(weights_ordered, annot=True, cmap='coolwarm', vmin=-0.5, vmax=0.5,
                    xticklabels=range(1, X_train.shape[1] + 1),
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


def filter_predictions(predictions, method='mean', window_size=3):
    """
    Apply a filter to the prediction values.

    Args:
    predictions (numpy.array): The original prediction values
    method (str): The filtering method to use ('mean' or 'median')
    window_size (int): The size of the sliding window for the filter

    Returns:
    numpy.array: The filtered prediction values
    """

    if method == 'mean':
        return uniform_filter1d(predictions, size=window_size)

    elif method == 'median':
        return median_filter(predictions, size=window_size)

    else:
        raise ValueError("Invalid method. Choose 'mean' or 'median'.")


def filter_and_rescale(predictions, filter_method='mean', filter_window=3,
                       rescale_range=(0, 1)):
    filtered = filter_predictions(predictions, method=filter_method, window_size=filter_window)
    min_val, max_val = rescale_range
    return min_val + (max_val - min_val) * (filtered - filtered.min()) / (
            filtered.max() - filtered.min())


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models for blendshape prediction")
    parser.add_argument("--data_path", default=f"C:/Users/YH006_new/fEMG_to_avatar/data", help="Path to data directory")
    parser.add_argument("--test_eeg", action="store_true", default=False, help="Test EEG flag")
    parser.add_argument("--ica_flag", action="store_true", default=True, help="Use ICA flag")
    parser.add_argument("--emg_flag", action="store_true", default=False, help="Use EMG flag")
    parser.add_argument("--train_one_trial", action="store_true", default=True)
    parser.add_argument("--trial_num", default='trial_1', choices=['trial_1', 'trial_2', 'trial_3'])
    parser.add_argument("--save_results", action="store_true", default=True, help="Save results flag")
    parser.add_argument("--scale_data", action="store_true", default=True, help="Scale data flag")
    parser.add_argument("--train_models", action="store_true", default=False, help="Train models or load parameters flag")
    parser.add_argument("--train_deep_learning_model", action="store_true", default=True,
                        help="Train deep learning model flag")
    parser.add_argument("--tune_autoencoder", action="store_true", default=False, help="Tune autoencoder flag")
    parser.add_argument("--train_linear_transform", action="store_true", default=True,
                        help="Train linear transform flag")
    parser.add_argument("--train_enhanced_linear_transform", action="store_true", default=True,
                        help="Train enhanced linear transform flag")
    parser.add_argument("--model_name", default='ImprovedEnhancedTransformNet', choices=['LinearTransformNet',
                                                                                 'EnhancedTransformNet',
                                                                                 'ImprovedEnhancedTransformNet'],
                        help="Model name")
    parser.add_argument("--train_autoencoder", action="store_true", default=False, help="Train autoencoder flag")
    parser.add_argument("--criterion", default='SmoothL1Loss', choices=['mse', 'pearson', 'SmoothL1Loss'], help="Loss criterion")
    parser.add_argument("--Early_stopping", action="store_true", default=True, help="Early stopping flag")
    parser.add_argument("--segments_length", type=int, default=4, help="Length of segments in seconds")
    parser.add_argument("--models", nargs='+',
                        default=['LR', 'ETR', 'Ridge', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor',
                                 'RandomForestRegressor'], help="Models to evaluate")

    args = parser.parse_args()

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    performance_results = []
    for participant_folder in os.listdir(args.data_path):
        if not participant_folder == 'participant_03':
            continue
        if 'csv' in participant_folder:
            continue
        participant_ID = participant_folder
        participant_folder_path = os.path.join(args.data_path, participant_folder)
        for session_folder in os.listdir(participant_folder_path):
            if not session_folder == 'S1':
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

                # Prepare data for model (existing code)
                annotations_list = ['05_Forehead', '07_Eye_gentle', '09_Eye_tight', '12_Nose', '14_Smile_closed',
                                    '16_Smile_open', '19_Lip_pucker', '21_Cheeks', '23_Snarl', '26_Depress_lip']
                annotations_list_to_remove = ([f"{annot}_trial_1" for annot in annotations_list] +
                                              [f"{annot}_trial_2" for annot in annotations_list] +
                                              [f"{annot}_trial_3" for annot in annotations_list])
                # print all the annotations that contains the strings in annotations_list
                annotations_list_with_start_end = []
                for annotation in emg_file.annotations.description:
                    if args.train_one_trial:
                        if args.trial_num in annotation:
                            if 'start' in annotation or 'end' in annotation:
                                if not 'Break' in annotation:
                                    if not 'Face_at_rest' in annotation:
                                        annotations_list_with_start_end.append(annotation)
                    else:
                        if 'start' in annotation or 'end' in annotation:
                            if not 'Break' in annotation:
                                if not 'Face_at_rest' in annotation:
                                    annotations_list_with_start_end.append(annotation)
                events_timings = get_annotations_timings(emg_file, annotations_list_with_start_end)
                # make events_timings into a list with 10 lists that each contains the start and end of the annotation
                events_timings = [[events_timings[i], events_timings[i + 1]] for i in range(0, len(events_timings), 2)]

                if args.ica_flag:
                    relevant_data_train_emg, relevant_data_test_emg, rand_lst, test_data_timing = prepare_relevant_data_new(
                        ica_after_order, emg_file, emg_fs, events_timings, False,
                        events_timings=events_timings,
                        segments_length=args.segments_length, norm="ICA",
                        averaging="RMS")
                elif args.emg_flag:
                    relevant_data_train_emg, relevant_data_test_emg, rand_lst, test_data_timing = prepare_relevant_data_new(
                        X_full, emg_file, emg_fs, events_timings, False,
                        events_timings=events_timings,
                        segments_length=args.segments_length, norm="ICA",
                        averaging="RMS")
                # Load avatar data (existing code)
                avatar_data = pd.read_csv(os.path.join(session_folder_path,
                                                       f"{participant_ID}_{session_number}_interpolated_relevant_only_right.csv"),
                                          header=0, index_col=0)
                blendshapes = avatar_data.columns
                relevant_data_train_avatar, relevant_data_test_avatar = prepare_avatar_relevant_data(participant_ID,
                                                                                                     avatar_data,
                                                                                                     emg_file,
                                                                                                     relevant_data_train_emg,
                                                                                                     relevant_data_test_emg,
                                                                                                     events_timings,
                                                                                                     False, rand_lst,
                                                                                                     fs=60,
                                                                                                     events_timings=events_timings,
                                                                                                     segments_length=args.segments_length,
                                                                                                     norm=None,
                                                                                                     averaging="RMS")
                # plot ICA components vs avatar blendshapes
                plot_ica_vs_blendshapes(annotations_list, test_data_timing, relevant_data_test_emg, 500,
                                        participant_ID,
                                        session_number, project_folder)
                X_train = relevant_data_train_emg.T
                X_test = relevant_data_test_emg.T
                Y_train = relevant_data_train_avatar.T
                Y_test = relevant_data_test_avatar.T

                if args.scale_data:
                    # # Standardize the features
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
                        model_name = args.model_name
                        path_to_best_params = fr"{project_folder}/results/best_params_{model_name}.json"
                        if args.train_one_trial:
                            path_to_best_params = path_to_best_params.replace(f'.json', f"_{args.trial_num}.json")
                        if args.train_models:
                            best_params = None
                        else:
                            best_params = load_best_params(path_to_best_params)
                        if best_params is None:
                            print("Best parameters not found. Running hyperparameter tuning...")
                            best_params, best_losses = tune_hyperparameters(X_train, Y_train, input_dim, output_dim,
                                                                            args.criterion, filename=path_to_best_params,
                                                                            model_name=model_name)
                            plot_loss_values(best_losses, best_params)


                        else:
                            print("Loaded best parameters:", best_params)
                        best_model, train_losses, test_losses, Y_pred = test_best_model(X_train, Y_train, X_test,
                                                                                        Y_test,
                                                                                        input_dim, output_dim,
                                                                                        best_params,
                                                                                        args.criterion, args.Early_stopping,
                                                                                        model_name=
                                                                                        model_name)
                        plot_model_performance(train_losses, test_losses)
                        #    save the model
                        model_path = os.path.join(session_folder_path,
                                                  f"{participant_ID}_{session_number}_blendshapes_{model_name}_{config}.joblib")
                        if args.train_one_trial:
                            model_path = model_path.replace(f'_{config}.joblib', f"_{args.trial_num}_{config}.joblib")
                        joblib.dump(best_model, model_path)
                        print(f"Model {model_name} for {config} saved as {model_path}")

                        # Move predictions back to CPU for further processing
                        Y_pred = Y_pred.cpu().numpy()
                        if args.save_results:
                            # perform the inverse transformation to get the original values
                            if args.scale_data:
                                Y_pred = scaler_Y.inverse_transform(Y_pred)
                            # filtered_and_rescaled = filter_and_rescale(Y_pred, filter_method='mean',
                            #                                            filter_window=3,
                            #                                            rescale_range=(0, 1))
                            # plot_ica_vs_blendshapes(annotations_list, test_data_timing, filtered_and_rescaled.T, 1,
                            #                         participant_ID, session_number)
                            pred_path = os.path.join(session_folder_path,
                                                     f"{participant_ID}_{session_number}_predicted_blendshapes_{model_name}_{config}.csv")
                            if args.train_one_trial:
                                pred_path = pred_path.replace(f'_{config}.csv', f"_{args.trial_num}_{config}.csv")
                            pd.DataFrame(Y_pred, columns=blendshapes).to_csv(pred_path)
                            print(f"Predicted data saved as {pred_path}")

                    if args.train_autoencoder:
                        print("Training Autoencoder Model...")
                        encoding_dim = output_dim  # You can adjust this

                        autoencoder = SimpleAutoencoder(input_dim, encoding_dim)

                        if args.tune_autoencoder:
                            # To tune batch size
                            _, best_batch_size, _ = train_autoencoder_cv(autoencoder, X_train.cpu().numpy(),
                                                                         args.criterion, tune_parameter='batch_size')

                            # To tune learning rate
                            _, _, best_lr = train_autoencoder_cv(autoencoder, X_train.cpu().numpy(), args.criterion,
                                                                 tune_parameter='learning_rate')
                        else:
                            best_batch_size = 16
                            best_lr = 0.01
                        print(f"Best batch size: {best_batch_size}, Best learning rate: {best_lr}")
                        # Train the best model
                        autoencoder, Y_pred = train_and_evaluate_best_model(autoencoder, X_train, X_test,
                                                                            args.criterion, best_batch_size, best_lr)

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
                            pred_path = os.path.join(session_folder_path,
                                                     f"{participant_ID}_{session_number}_predicted_blendshapes_{model_name}_{config}.csv")
                            if args.train_one_trial:
                                pred_path = pred_path.replace(f'_{config}.csv', f"_{args.trial_num}_{config}.csv")
                            pd.DataFrame(Y_pred, columns=blendshapes).to_csv(pred_path)
                            model_path = os.path.join(session_folder_path,
                                                      f"{participant_ID}_{session_number}_blendshapes_model_{model_name}_{config}.joblib")
                            joblib.dump(autoencoder, model_path)
                            print(f"Model {model_name} for {config} saved as {model_path}")

                else:
                    # Run models
                    for model_name in args.models:
                        model_path = os.path.join(session_folder_path,
                                                  f"{participant_ID}_{session_number}_blendshapes_model_{model_name}_{config}.joblib")

                        if args.emg_flag:
                            model, mse, r2, mae, Y_pred = evaluate_models(X_train, X_test, Y_train, Y_test,
                                                                          model_name,
                                                                          f'{participant_ID} {session_number} {config}',
                                                                          is_ICA=False)
                        else:
                            model, mse, r2, mae, Y_pred = evaluate_models(X_train, X_test, Y_train, Y_test,
                                                                          model_name,
                                                                          f'{participant_ID} {session_number} {config}',
                                                                          is_ICA=True)

                        if args.save_results:
                            # perform the inverse transformation to get the original values
                            Y_pred = scaler_Y.inverse_transform(Y_pred)
                            pred_path = os.path.join(session_folder_path,
                                                     f"{participant_ID}_{session_number}_predicted_blendshapes_{model_name}_{config}.csv")
                            pd.DataFrame(Y_pred, columns=blendshapes).to_csv(pred_path)
                            print(f"Predicted data saved as {pred_path}")

                            # Save model
                            joblib.dump(model, model_path)
                            print(f"Model {model_name} for {config} saved as {model_path}")

                    # After running both configurations, save performance results
                    performance_df = pd.DataFrame(performance_results)
                    performance_csv_path = os.path.join(session_folder_path,
                                                        f"{participant_ID}_{session_number}_model_performance_comparison.csv")
                    performance_df.to_csv(performance_csv_path, index=False)
                    print(f"Model performance comparison results saved to {performance_csv_path}")

                    # Reset performance_results for the next session
                    performance_results = []

                plot_predictions_vs_avatar(Y_pred, relevant_data_test_avatar.T, blendshapes, annotations_list,
                                           test_data_timing, project_folder)
                plot_correlations_barplot(Y_pred, relevant_data_test_avatar.T, project_folder)
                # convert the test values to the original scale
                Y_test = relevant_data_test_avatar.T
                # Save avatar data (existing code)
                avatar_sliding_window_method = "RMS"
                path = os.path.join(session_folder_path,

                                 f"{participant_ID}_{session_number}_avatar_blendshapes_{avatar_sliding_window_method}.csv")
                if args.train_one_trial:
                    path = path.replace(f'.csv', f"_{args.trial_num}.csv")
                pd.DataFrame(Y_test, columns=blendshapes).to_csv(path)

                print("Avatar data saved as CSV file.\n")


def plot_correlations_barplot(Y_pred, Y_test, project_folder):
    """
    Create a horizontal barplot of correlations between predicted and ground truth values for each action unit,
    with proper bar spacing and visibility of negative values.
    """
    # Calculate correlations for all action units
    correlations = []
    for i in range(Y_test.shape[1]):
        corr, _ = pearsonr(Y_pred[i], Y_test[i])
        correlations.append(corr)

    # Create figure
    plt.figure(figsize=(10, 12), dpi=300)
    plt.rcParams.update({'font.size': 26})

    # Create horizontal barplot
    y_pos = np.arange(len(correlations)) * 1.2  # Multiply by 1.2 for spacing between bars
    bars = plt.barh(y_pos, correlations, height=0.8)  # Height < 1 creates space between bars

    # Customize bars with colors
    for idx, bar in enumerate(bars):
        if correlations[idx] < 0:
            bar.set_color('red')
        else:
            bar.set_color('#4989E5')  # Light blue color

    # Customize plot
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Set labels
    plt.xlabel('Correlation Coefficient', fontsize=34)
    plt.ylabel('Action Unit', fontsize=34)

    # Add grid
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)

    # Set x-axis limits to show negative values
    min_corr = min(min(correlations) - 0.1, -0.1)  # Ensure we show at least to -0.1
    max_corr = max(max(correlations) + 0.1, 1.0)   # Ensure we show at least to 1.0
    plt.xlim(min_corr, max_corr)

    # Set y-axis limits and ticks
    plt.yticks(y_pos, range(1, len(correlations) + 1), fontsize=24)
    plt.ylim(min(y_pos) - 0.6, max(y_pos) + 0.6)  # Adjust limits to remove extra space

    # Move x-axis to top
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Remove top and right spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add more space at the top for labels
    plt.subplots_adjust(top=0.85, left=0.15, right=0.95)

    # Save plot
    plt.savefig(fr"{project_folder}\results\correlations_barplot.png", bbox_inches='tight')
    plt.close()



def plot_predictions_vs_avatar(Y_pred, Y_test, blendshapes, annotations_list, test_data_timing, project_folder):
    # Calculate correlations for each action unit
    correlations = []
    for i in range(Y_test.shape[1]):
        corr, _ = pearsonr(Y_pred[i], Y_test[i])
        correlations.append((i, corr))

    # Sort correlations in descending order and get top 10
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_16 = correlations[:16]
    num_plots = len(correlations[:16])
    fig, axs = plt.subplots(16, 1, figsize=(16, 26), dpi=300, sharex=True)
    plt.rcParams.update({'font.size': 32})

    # Calculate the overall time range for both EMG and avatar data
    max_time_emg = max(len(data) for data in Y_pred)
    max_time_avatar = max(len(data) for data in Y_test)
    max_time = max(max_time_emg, max_time_avatar)

    # Find global min and max for both predicted and ground truth
    data_min = min(min(np.min(Y_pred[i]), np.min(Y_test[i])) for i in range(Y_test.shape[1]))
    data_max = max(max(np.max(Y_pred[i]), np.max(Y_test[i])) for i in range(Y_test.shape[1]))

    # Process annotations
    annotations_list_edited = [annot[2:].replace("_", " ")+"  " for annot in annotations_list]
    annotation_positions = [sum(int(test_data_timing[j][1] - test_data_timing[j][0]) // 1.26 for j in range(i)) for i in
                            range(len(test_data_timing))]
    annotation_positions.append(max_time)  # Add the last position

    for plot_index, (i, corr) in enumerate(top_16):
        ax = axs[plot_index] if num_plots > 1 else axs  # Handle case when there's only one subplot

        time_axis_pred = np.arange(len(Y_pred[i]))
        time_axis_test = np.arange(len(Y_test[i]))

        # Increased linewidth for both plots
        pred_line, = ax.plot(time_axis_pred, Y_pred[i], color='blue', linewidth=6)
        truth_line, = ax.plot(time_axis_test, Y_test[i], color='red', linestyle='--', linewidth=6)

        # Only add legend to the first subplot
        fig.legend([pred_line, truth_line], ['Predicted', 'Ground Truth'],
                   loc='upper center', bbox_to_anchor=(0.5, 1),
                   ncol=2, fontsize=30)

        ax.set_ylim(data_min, data_max)
        ax.set_xlim(0, max_time)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.7)

        # Adjust tick parameters
        ax.tick_params(axis='both', which='both', labelsize=32)  # Reduced tick label size

        # Add y-axis label with AU number and correlation
        ax.set_ylabel(f'AU {i + 1}', rotation=0, ha='right', va='center', fontsize=32)

        # Adjust y-axis label position
        ax.yaxis.set_label_coords(-0.1, 0.5)  # Move y-label to the left

        # Add vertical lines for annotations
        for pos in annotation_positions[:-1]:
            ax.axvline(x=pos, color='red', linestyle='--', linewidth=0.7, alpha=0.7)

    # Set x-axis ticks and labels
    last_ax = axs[-1] if num_plots > 1 else axs
    last_ax.set_xticks(annotation_positions[:-1])
    last_ax.set_xticklabels(annotations_list_edited, rotation=90, ha='center')

    plt.tight_layout(rect=[0.05, 0.03, 1, 0.98])  # Adjust margins
    plt.savefig(fr"{project_folder}\results\predictions_vs_avatar_top16_corr.png")
    plt.close()


def plot_ica_vs_blendshapes(annotations_list, test_data_timing, relevant_data_test_emg, emg_fs, participant_ID,
                            session_number, project_folder):
    # Process annotations
    annotations_list_edited = [annot[2:].replace("_", " ") + "  " for annot in annotations_list]
    num_plots = len(relevant_data_test_emg)
    fig, axs = plt.subplots(num_plots, 1, figsize=(16, 26), dpi=300, sharex=True)
    plt.rcParams.update({'font.size': 32})

    # Normalize ICA data
    def normalize_data(data):
        return (data - np.mean(data)) / np.std(data)

    normalized_emg = [normalize_data(data) for data in relevant_data_test_emg]
    # Calculate the overall time range for EMG data
    max_time_emg = max(len(data) for data in normalized_emg) / emg_fs
    # Calculate annotation positions similar to first code
    annotation_positions = [sum(0.4 + test_data_timing[j][1] - test_data_timing[j][0] for j in range(i)) for i in
                            range(len(test_data_timing))]
    annotation_positions.append(max_time_emg)  # Add the last position
    # Find global min and max for normalized data
    data_min = min(np.min(data) for data in normalized_emg)
    data_max = max(np.max(data) for data in normalized_emg)
    for i in range(num_plots):
        # Reverse the order of ICA plots
        ica_index = num_plots - i - 1
        ax = axs[i] if num_plots > 1 else axs
        time_axis_emg = np.arange(len(normalized_emg[ica_index])) / emg_fs
        # Plot with increased linewidth to match first code
        ica_line = ax.plot(time_axis_emg, normalized_emg[ica_index], linewidth=0.8, color='blue')[0]
        # Set consistent y-axis limits based on data
        ax.set_ylim(-9, 9)
        ax.set_xlim(0, max_time_emg)
        ax.set_yticks([-5, 5])

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.7)
        # Adjust tick parameters to match first code
        ax.tick_params(axis='both', which='both', labelsize=32)
        # Set y-axis label with consistent positioning
        ax.set_ylabel(f'ICA {ica_index + 1}', rotation=0, ha='right', va='center', fontsize=32)
        # ax.yaxis.set_label_coords(-0.1, 0.5)  # Match the position from first code
        # Add vertical lines for annotations
        for pos in annotation_positions[:-1]:
            ax.axvline(x=pos, color='red', linestyle='--', linewidth=0.7, alpha=0.7)
    # Set x-axis ticks and labels for bottom subplot
    last_ax = axs[-1] if num_plots > 1 else axs
    last_ax.set_xticks(annotation_positions[:-1])
    last_ax.set_xticklabels(annotations_list_edited, rotation=90, ha='center')
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.98])  # Adjust margins to match first code
    plt.savefig(fr"{project_folder}\results\{participant_ID}_{session_number}_ICA_components.png")
    plt.close()


if __name__ == "__main__":
    main()
