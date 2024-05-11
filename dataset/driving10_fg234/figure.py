import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载原始速度数据
df = pd.read_csv('client_4_vehicle.audi.a2_202403072232.csv')
# 加载预测结果
predictions = np.load('predictions.npy')

def plot_predictions_with_interval(original_velocities, predictions, start, end, window_size=10, prediction_length=10, interval=20):

    plt.figure(figsize=(15, 5))
    plt.plot(range(start, end), original_velocities[start:end], label='Original Velocity', color='blue', linewidth=2.5)
    if start > window_size:
        prediction_start = start - window_size + 1
    else:
        prediction_start = start
    for i in range(prediction_start, end - window_size - prediction_length + 1, interval):
        prediction_start_time = i + window_size
        if prediction_start_time + prediction_length <= end:
            # 获取与预测起始时间对应的实际速度值
            actual_start_velocity = original_velocities[prediction_start_time - 1]  # -1 是因为需要前1s的值
            # 构建预测曲线的时间范围和速度值
            prediction_time_range = [prediction_start_time - 1] + list(range(prediction_start_time, prediction_start_time + prediction_length))
            prediction_velocities = [actual_start_velocity] + list(predictions[i][:prediction_length])
            # 绘制预测曲线
            plt.plot(prediction_time_range, prediction_velocities, marker='s', markevery=2 , color='red', linewidth=2.5)


    # 设置标题、图例和坐标轴标签
    plt.title('Vehicle Speed Prediction with Interval', fontsize=18, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
    plt.ylabel('Velocity (units)', fontsize=14, fontweight='bold')
    plt.legend(['Original Velocity', 'Predicted Velocity'], fontsize=12, loc='upper right')

    # 美化图表
    plt.grid(True)
    plt.xlim(start,end)  # 限制X轴的范围
    plt.tight_layout()
    # 显示图表
    plt.show()

# 示例调用绘图函数
plot_predictions_with_interval(df['velocity'].values, predictions, 1000, 1500, interval=15)










