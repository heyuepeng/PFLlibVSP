import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


plt.rcParams.update({
    'font.size': 16,  # 全局字体大小适中
    'xtick.labelsize': 16,  # X轴刻度标签字体大小
    'ytick.labelsize': 16,  # Y轴刻度标签字体大小
    'axes.labelsize': 18,  # 轴标签字体大小
})

# 初始化一个空的DataFrame用于存储所有client的数据
all_data = pd.DataFrame()

# 文件夹路径
data_folder = 'driving/rawdata'

# 遍历文件夹读取数据
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        # 提取client ID
        client_id = int(file.split('_')[1])  # 将client_id转换为整数，以便排序

        # 读取CSV文件
        filepath = os.path.join(data_folder, file)
        df = pd.read_csv(filepath)
        # 清除NaN值
        df = df.dropna(subset=['velocity'])
        # 将velocity转换为其绝对值
        df['velocity'] = df['velocity'].abs()
        # 丢弃小于0.01或大于30的velocity值
        df = df[(df['velocity'] > 0.01) & (df['velocity'] < 30)]
        # 添加client_id列，这里存储为整数便于后续排序
        df['client_id'] = client_id
        # 合并数据
        all_data = pd.concat([all_data, df[['velocity', 'client_id']]], ignore_index=True)

# 对client_id进行排序
all_data.sort_values('client_id', inplace=True)


# 绘制箱型图，调整异常值的判定范围
plt.figure(figsize=(12, 6))
boxplot = all_data.boxplot(by='client_id', column=['velocity'], grid=True, showfliers=False, whis=5,
                           patch_artist=True,
                           boxprops=dict(facecolor='#8FAADC', color='black'),  # 箱体为蓝色，边框为黑色
                           medianprops=dict(color='black'),  # 中位数线为黑色
                           whiskerprops=dict(color='black'),  # 须线为黑色
                           capprops=dict(color='black'))  # 须帽为黑色

# 美化图形
plt.xlabel('Client ID')
plt.ylabel('Vehicle Speed (m/s)')  # 添加单位

ax = plt.gca()
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.grid(linestyle='-', color='lightgray')  # 设置网格线更浅
plt.suptitle('')  # 移除默认的副标题
plt.title('')  # 确保没有标题

plt.tight_layout()
# 保存为PDF格式
plt.savefig('velocity_distribution.pdf', bbox_inches='tight', format='pdf', dpi=1000)

plt.show()