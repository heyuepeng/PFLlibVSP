import pandas as pd
import numpy as np
import torch
import joblib

# 加载数据
df = pd.read_csv('client_4_vehicle.audi.a2_202403141252.csv')
features = ['velocity', 'has_front_vehicle', 'front_vehicle_distance', 'front_vehicle_velocity',
            'throttle', 'steer', 'brake',
            'car_pixels_ratio', 'truck_bus_pixels_ratio', 'traffic_light_pixels_ratio',
            'has_traffic_light', 'distance_tl', 'signal_tl', 'signal_tl_1s', 'signal_tl_2s', 'signal_tl_3s',
            'signal_tl_4s', 'signal_tl_5s']

# 加载归一化器
scaler_features = joblib.load('scaler_features.pkl')
scaler_target = joblib.load('scaler_target.pkl')

# 数据归一化
features_scaled = scaler_features.transform(df[features])

# 定义预处理函数
def preprocess_data(features_scaled, window_size=5):
    sequences = [features_scaled[i:i+window_size] for i in range(len(features_scaled)-window_size+1)]
    return np.array(sequences, dtype=np.float32)

# 预处理数据
input_sequences = preprocess_data(features_scaled)

# 加载模型
model = torch.load('FedAWA_personalized_model_3.pt', map_location=torch.device('cpu'))
model.eval()

# 预测
predictions = []
with torch.no_grad():
    for sequence in input_sequences:
        sequence = torch.tensor(sequence[np.newaxis, :], dtype=torch.float32)  # 添加批次维度
        prediction = model(sequence).numpy().squeeze()  # 移除批次维度
        prediction = scaler_target.inverse_transform(prediction.reshape(-1, 1)).squeeze()  # 反归一化
        predictions.append(prediction)

# 将预测结果保存到文件
np.save('predictions_fedpaw.npy', np.array(predictions))
