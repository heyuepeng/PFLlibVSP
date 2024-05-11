import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载原始速度数据
df = pd.read_csv('client_4_vehicle.audi.a2_202403141252.csv')
# 加载预测结果
predictions = np.load('predictions_fedpaw.npy')

# 调整全局字体和图例设置
plt.rcParams.update({
    'font.size': 20,  # 全局字体大小适中
    'xtick.labelsize': 20,  # X轴刻度标签字体大小
    'ytick.labelsize': 20,  # Y轴刻度标签字体大小
    'legend.fontsize': 18,  # 图例字体大小
    'axes.labelsize': 22,  # 轴标签字体大小
    'legend.edgecolor': 'black',  # 图例边框颜色
    'legend.framealpha': 1,  # 图例边框透明度
    'legend.frameon': True,  # 开启图例边框
})

def plot_predictions_with_interval(original_velocities, predictions, start, end, window_size=5, prediction_length=5, interval=10):

    plt.figure(figsize=(18, 5))
    plt.plot(range(start, end), original_velocities[start:end], color='#0055D4', linewidth=3)
    if start > window_size:
        prediction_start = start - window_size + 3
    else:
        prediction_start = start
    for i in range(prediction_start, end - window_size - prediction_length + 1, interval):
        prediction_start_time = i + window_size
        if prediction_start_time + prediction_length < end:
            # 获取与预测起始时间对应的实际速度值
            actual_start_velocity = original_velocities[prediction_start_time - 1]  # -1 是因为需要前1s的值
            # 构建预测曲线的时间范围和速度值
            prediction_time_range = [prediction_start_time - 1] + list(range(prediction_start_time, prediction_start_time + prediction_length))
            prediction_velocities = [actual_start_velocity] + list(predictions[i][:prediction_length])
            # 绘制预测曲线
            plt.plot(prediction_time_range, prediction_velocities, 'r--', linewidth=1)

    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')

    legend = plt.legend(['Actual Speed', 'FedPAW'], fontsize=16, loc='upper center',
                        bbox_to_anchor=(0.52, 1), frameon=True)
    legend.get_frame().set_linewidth(1.2)  # 加粗图例边框

    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)

    plt.grid(True)
    plt.xlim(start, end)
    plt.tight_layout()
    plt.savefig('prediction_v.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.show()

# 示例调用绘图函数
plot_predictions_with_interval(df['velocity'].values, predictions, 1300, 1800, interval=1)










