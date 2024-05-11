import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import json

def clean_data(df):
    # Remove rows with NaN in any column
    df.dropna(inplace=True)
    # Ensure velocity is non-negative
    df['velocity'] = df['velocity'].abs()
    return df

def merge_client_data(data_dir, client_id, features):
    data_path = os.path.join(data_dir, 'rawdata')
    client_files = sorted(
        [f for f in os.listdir(data_path) if f.startswith(f"client_{client_id}_") and f.endswith(".csv")])
    client_dfs = []
    for f in client_files:
        df = pd.read_csv(os.path.join(data_path, f))
        df = clean_data(df)  # Clean the data
        client_dfs.append(df)
    client_df = pd.concat(client_dfs, ignore_index=True)
    return client_df[features], client_df['velocity'].values.reshape(-1, 1)

def preprocess_data(features, target, scaler_features, scaler_target, window_size, prediction_window):
    features_scaled = scaler_features.transform(features)
    target_scaled = scaler_target.transform(target)
    input_seq, output_seq = [], []
    for i in range(len(features_scaled) - window_size - prediction_window + 1):
        input_seq.append(features_scaled[i:i+window_size])
        output_seq.append(target_scaled[i+window_size:i+window_size+prediction_window])  #  [prediction_window, ]
    return np.array(input_seq, dtype=np.float32), np.array(output_seq, dtype=np.float32)


def generate_global_scalers(data_dir, features):
    data_path = os.path.join(data_dir, 'rawdata')
    all_features = []
    all_targets = []
    for client_id in range(1, 11):  # Assuming client IDs are from 1 to 10
        client_files = [f for f in os.listdir(data_path) if f.startswith(f"client_{client_id}_") and f.endswith(".csv")]
        for client_file in client_files:
            file_path = os.path.join(data_path, client_file)
            df = pd.read_csv(file_path)
            df = clean_data(df)  # Clean the data
            all_features.append(df[features])
            all_targets.append(df['velocity'].values.reshape(-1, 1))
    all_features_df = pd.concat(all_features, ignore_index=True)
    all_targets = np.concatenate(all_targets, axis=0)
    scaler_features = MinMaxScaler().fit(all_features_df)
    scaler_target = MinMaxScaler().fit(all_targets)
    return scaler_features, scaler_target

def save_data(data_dir, X, y, dataset_name, client_id):
    data_path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # 保存X和y在名为'data'的字典内
    np.savez_compressed(os.path.join(data_path, f"{client_id-1}.npz"), data={'x': X, 'y': y})


def generate_client_data(data_dir, features, window_size, prediction_window):
    scaler_features, scaler_target = generate_global_scalers(data_dir, features)
    statistics = []
    for client_id in range(1, 11):  # Assuming client IDs are from 1 to 10
        merged_features, merged_target = merge_client_data(data_dir, client_id, features)
        X, y = preprocess_data(merged_features, merged_target, scaler_features, scaler_target, window_size,
                               prediction_window)

        # 按顺序划分训练集和测试集，后20%为测试集
        test_size = int(len(X) * 0.8)  # 计算80%的位置
        X_train, y_train = X[:test_size], y[:test_size]
        X_test, y_test = X[test_size:], y[test_size:]

        save_data(data_dir, X_train, y_train, "train", client_id)
        save_data(data_dir, X_test, y_test, "test", client_id)
        statistics.append({'client_id': client_id, 'train_samples': len(X_train), 'test_samples': len(X_test)})

    # Save scalers at the client directory level
    joblib.dump(scaler_features, os.path.join(data_dir, "scaler_features.pkl"))
    joblib.dump(scaler_target, os.path.join(data_dir, "scaler_target.pkl"))

    # Saving configuration
    config = {'num_clients': 10, 'statistics': statistics, 'window_size': window_size,
              'prediction_window': prediction_window}
    with open(os.path.join(data_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    data_dir = "./driving10_fg234"
    features = ['velocity', 'has_front_vehicle', 'front_vehicle_distance', 'front_vehicle_velocity',
                'has_side_vehicle', 'side_vehicle_distance', 'side_vehicle_velocity', 'throttle', 'steer', 'brake',
                'car_pixels_ratio', 'truck_bus_pixels_ratio', 'traffic_light_pixels_ratio',
                'has_traffic_light', 'distance_tl', 'signal_tl',
                'signal_tl_1s', 'signal_tl_2s', 'signal_tl_3s', 'signal_tl_4s', 'signal_tl_5s', 'signal_tl_6s', 'signal_tl_7s', 'signal_tl_8s', 'signal_tl_9s', 'signal_tl_10s']
    generate_client_data(data_dir, features, 10, 10)
    print("Finish generating driving10 dataset.\n")
