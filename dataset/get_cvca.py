import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Constants
FUTURE_PREDICTION_STEPS = 5  # Predicting 5 seconds into the future
rawdata_path = 'driving10/rawdata/'

# Revised functions for the CV and CA models based on the user's correction

def constant_velocity_model(velocity_data, t_minus_2, t_minus_1):
    # CV model uses the average of the last two velocity readings to predict the next velocity
    v_t_bar = (velocity_data[t_minus_1] + velocity_data[t_minus_2]) / 2
    return v_t_bar


def constant_acceleration_model(velocity_data, t_minus_2, t_minus_1):
    # CA model uses the acceleration (difference in velocity / time) to predict the next velocity
    a_t = (velocity_data[t_minus_1] - velocity_data[t_minus_2])
    v_t = velocity_data[t_minus_1] + a_t
    return v_t

def calculate_errors(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

# Initialize lists to store MAE and RMSE for each 5-second prediction for all CSV files
all_mae_cv = []
all_rmse_cv = []
all_mae_ca = []
all_rmse_ca = []

# Loop through each CSV file in the 'rawdata' directory
for csv_file in os.listdir(rawdata_path):
    file_path = os.path.join(rawdata_path, csv_file)
    velocity_df = pd.read_csv(file_path)

    # Ensure 'velocity' column exists and drop rows with NaN in 'velocity'
    if 'velocity' in velocity_df.columns:
        velocity_data = velocity_df['velocity'].dropna().reset_index(drop=True)

    # Initialize lists to store the predictions and actual values for this file
    cv_predictions = []
    ca_predictions = []
    actual_values = []

    # Loop over the velocity data in 5-second windows
    for i in range(2, len(velocity_data) - FUTURE_PREDICTION_STEPS):
        # Skip if any of the involved velocities are NaN
        if np.isnan(velocity_data[i - 2]) or np.isnan(velocity_data[i - 1]) or np.any(np.isnan(velocity_data[i + 1:i + 1 + FUTURE_PREDICTION_STEPS])):
            continue
        cv_pred_window = []
        ca_pred_window = []
        actual_window = velocity_data[i + 1:i + 1 + FUTURE_PREDICTION_STEPS]  # Actual velocities for the next 5 seconds

        # Use CV and CA models to predict the next 5 seconds based on the last 2 seconds
        for j in range(FUTURE_PREDICTION_STEPS):
            cv_pred = constant_velocity_model(velocity_data, i - 2, i - 1)
            ca_pred = constant_acceleration_model(velocity_data, i - 2, i - 1)
            cv_pred_window.append(cv_pred)
            ca_pred_window.append(ca_pred)
            # Update the velocity data for prediction using the CA model

        # Store predictions and actual values
        cv_predictions.append(cv_pred_window)
        ca_predictions.append(ca_pred_window)
        actual_values.append(actual_window)

        # Calculate errors for this 5-second window and store them
        mae_cv, rmse_cv = calculate_errors(np.array(actual_window), np.array(cv_pred_window))
        mae_ca, rmse_ca = calculate_errors(np.array(actual_window), np.array(ca_pred_window))
        all_mae_cv.append(mae_cv)
        all_rmse_cv.append(rmse_cv)
        all_mae_ca.append(mae_ca)
        all_rmse_ca.append(rmse_ca)

# Calculate the average MAE and RMSE across all 5-second windows for all CSV files
average_mae_cv = np.mean(all_mae_cv)
average_rmse_cv = np.mean(all_rmse_cv)
average_mae_ca = np.mean(all_mae_ca)
average_rmse_ca = np.mean(all_rmse_ca)

print("average_mae_cv:",average_mae_cv)
print("average_rmse_cv:",average_rmse_cv)
print("average_mae_ca:",average_mae_ca)
print("average_rmse_ca:",average_rmse_ca)

