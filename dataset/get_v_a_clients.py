import os
import pandas as pd

# 定义一个字典来存储每个客户的速度和加速度数据
client_data = {}

# 遍历driving/rawdata文件夹中的所有CSV文件
data_folder = 'driving/rawdata'
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        # 提取客户ID
        client_id = file.split('_')[1]

        # 读取CSV文件
        df = pd.read_csv(os.path.join(data_folder, file))

        # 清除NaN值
        df = df.dropna(subset=['velocity', 'acceleration'])

        # 筛选并计算平均速度和加速度的绝对值
        avg_velocity = df['velocity'].abs()[df['velocity'].abs() > 0.01].mean()
        avg_acceleration = df['acceleration'].abs()[df['acceleration'].abs() > 0.01].mean()

        # 存储数据
        if client_id not in client_data:
            client_data[client_id] = {'velocity': [], 'acceleration': []}
        client_data[client_id]['velocity'].append(avg_velocity)
        client_data[client_id]['acceleration'].append(avg_acceleration)

# 准备数据以保存到CSV
client_ids = sorted(client_data, key=int)
avg_velocities = [sum(client_data[client_id]['velocity']) / len(client_data[client_id]['velocity']) for client_id in client_ids]
avg_accelerations = [sum(client_data[client_id]['acceleration']) / len(client_data[client_id]['acceleration']) for client_id in client_ids]

# 创建DataFrame
df_results = pd.DataFrame({
    'Client ID': client_ids,
    'Avg Velocity': avg_velocities,
    'Avg Acceleration': avg_accelerations
})

# 按照Client ID排序（虽然已经是排序的，这里确保无误）
# df_results.sort_values('Client ID', inplace=True)

# 保存为CSV
df_results.to_csv('client_driving_data_summary.csv', index=False)

print("数据已保存为CSV格式，文件名为: client_driving_data_summary.csv")
