import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

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
        df = df.dropna(subset=['acceleration'])

        # 过滤极端的加速度值
        df = df[(df['acceleration'] > -10) & (df['acceleration'] < 8)]

        # 过滤掉加速度绝对值非常小的数据点，例如小于0.1m/s^2的
        df = df[(df['acceleration'].abs() > 0.1) & (df['velocity'].abs() > 0.1)]

        # 添加client_id列，这里存储为整数便于后续排序
        df['client_id'] = client_id

        # 合并数据
        all_data = pd.concat([all_data, df[['acceleration', 'client_id']]], ignore_index=True)

# 对client_id进行排序
all_data.sort_values('client_id', inplace=True)

# 设置matplotlib字体为加粗的Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'

# 绘制箱型图，调整异常值的判定范围
plt.figure(figsize=(12, 6))
boxplot = all_data.boxplot(by='client_id', column=['acceleration'], grid=True, showfliers=False, whis=15,
                           patch_artist=True,
                           boxprops=dict(facecolor='#8FAADC', color='black'),
                           medianprops=dict(color='black'),
                           whiskerprops=dict(color='black'),
                           capprops=dict(color='black'))

# 美化图形
plt.xlabel('Client ID', fontsize=12)
plt.ylabel('Vehicle Acceleration (m/s^2)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(linestyle='-', linewidth='0.5', color='lightgray')
plt.suptitle('')
plt.title('')

# 保存为PDF格式
plt.savefig('acceleration_distribution.pdf', bbox_inches='tight', format='pdf', dpi=1000)

plt.show()
