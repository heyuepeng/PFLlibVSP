import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载原始速度数据
df = pd.read_csv('client_4_vehicle.audi.a2_202403141252.csv')
# 加载预测结果
predictions_fedpaw = np.load('predictions_fedpaw.npy')
predictions_fedavg = np.load("predictions_fedavg.npy")

# 调整全局字体和图例设置
plt.rcParams.update({
    'font.size': 20,  # 全局字体大小适中
    'xtick.labelsize': 20,  # X轴刻度标签字体大小
    'ytick.labelsize': 20,  # Y轴刻度标签字体大小
    'legend.fontsize': 18,  # 图例字体大小
    'axes.labelsize': 24,  # 轴标签字体大小
    'legend.edgecolor': 'black',  # 图例边框颜色
    'legend.framealpha': 1,  # 图例边框透明度
    'legend.frameon': True,  # 开启图例边框
})
def constant_velocity_model(velocity_data, t_minus_1, t_minus_2 , prediction_length):
    # CV模型：使用当前速度预测未来速度
    v_t = (velocity_data[t_minus_1] + velocity_data[t_minus_2]) / 2
    return [velocity_data[t_minus_1]] + [v_t] * prediction_length  # 包含当前速度和后续5秒的速度预测

def constant_acceleration_model(velocity_data, t_minus_1, t_minus_2, delta_t, prediction_length):
    # CA模型：根据当前加速度预测未来速度
    a_t = (velocity_data[t_minus_1] - velocity_data[t_minus_2]) / delta_t
    return [velocity_data[t_minus_1]] + [velocity_data[t_minus_1] + a_t * delta_t * i for i in range(1, prediction_length + 1)]

def plot_predictions_with_interval(original_velocities, predictions_fedpaw, predictions_fedavg, start, end, window_size=5,
                                   prediction_length=5, prediction_points=[]):
    plt.figure(figsize=(10, 6))
    # 实际速度曲线
    plt.plot(range(start, end + 1), original_velocities[start:end + 1], color='#0055D4', linewidth=3,
             label='Actual Speed')

    for i in prediction_points:
        i_adjusted = i - window_size + 1
        prediction_start_time = i_adjusted + window_size
        if prediction_start_time + prediction_length <= end:
            actual_start_velocity = original_velocities[prediction_start_time - 1]
            prediction_time_range = [prediction_start_time - 1] + list(
                range(prediction_start_time, prediction_start_time + prediction_length))

            cv_predictions = constant_velocity_model(original_velocities, prediction_start_time - 1, prediction_start_time - 2, prediction_length)
            ca_predictions = constant_acceleration_model(original_velocities, prediction_start_time - 1, prediction_start_time - 2, 1, prediction_length)

            plt.plot(prediction_time_range, cv_predictions, color='black', linestyle='--', linewidth=2.5,
                     label='CV Model' if i == prediction_points[0] else "")
            plt.plot(prediction_time_range, ca_predictions, color='#ff7f0e', linestyle='-.', linewidth=2.5,
                     label='CA Model' if i == prediction_points[0] else "")

            # FedAvg预测曲线
            prediction_velocities_avg = [actual_start_velocity] + list(
                predictions_fedavg[i_adjusted][:prediction_length])
            plt.plot(prediction_time_range, prediction_velocities_avg, color= '#2ca02c', linewidth=2, marker='o', markersize=8)
            # FedPAW预测曲线
            prediction_velocities_paw = [actual_start_velocity] + list(predictions_fedpaw[i_adjusted][:prediction_length])
            plt.plot(prediction_time_range, prediction_velocities_paw, color='#ff0000', linewidth=2, marker='s', markersize=8)


    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')

    # 单独添加图例，确保所有曲线都被表示
    legend = plt.legend(['Actual Speed', 'CV', 'CA', 'FedAvg', 'FedPAW'], fontsize=16, frameon=True)
    legend.get_frame().set_linewidth(1.2)  # 加粗图例边框
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.grid(True)
    plt.xlim(start, end)
    plt.tight_layout()
    plt.savefig('prediction_v_part.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.show()


# 调用绘图函数时指定预测点
specific_prediction_points = [1635, 1645, 1653, 1664, 1671]  # 这些点应该基于你的实际需求进行选择
plot_predictions_with_interval(df['velocity'].values, predictions_fedpaw, predictions_fedavg, 1630, 1680,
                               prediction_points=specific_prediction_points)
