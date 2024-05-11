import h5py
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
plt.rcParams.update({
    'font.size': 36,  # 调整全局字体大小
    'xtick.labelsize': 36,
    'ytick.labelsize': 36,
    'legend.fontsize': 34,
    'axes.labelsize': 40,  # 调整轴标签字体大小
    'legend.edgecolor': 'black',  # 设置图例边框颜色
    'legend.framealpha': 1,  # 设置图例边框透明度
    'legend.frameon': True,  # 开启图例边框
})

# HDF5文件列表和算法名称
hdf5_files = [
    'driving_fg34_FedAvg_test_2.h5', 'driving_fg34_FedProx_test_2.h5', 'driving_fg34_PerAvg_test_2.h5',
    'driving_fg34_pFedMe_test_2.h5', 'driving_fg34_Ditto_test_2.h5', 'driving_fg34_APFL_test_2.h5',
    'driving_fg34_FedFomo_test_0.h5', 'driving_fg34_FedALA_test_4.h5', 'driving_fg34_FedAWA_test_4.h5'
]
names = ['FedAvg', 'FedProx', 'Per-FedAvg', 'pFedMe', 'Ditto', 'APFL', 'FedFomo', 'FedALA', 'FedPAW']

colors = ['#1f77b4', '#2ca02c', '#9467bd', '#e377c2', '#bcbd22', '#7f7f7f', '#ff7f0e', '#17becf', '#ff0000']

linestyle = '-'
markers = ['o', 'v', 'P', '*', '>', 'H', '+', 'x', 's']

plt.figure(figsize=(16, 12))

for hdf5_file, name, marker, color in zip(hdf5_files, names, markers, colors):
    with h5py.File(hdf5_file, 'r') as hf:
        rs_test_mae = hf['rs_test_mae'][:201]

    plt.plot(rs_test_mae, label=name, linestyle=linestyle, marker=marker, markevery=5, linewidth=3, markersize=10, color=color)

# 加粗坐标轴线条
ax = plt.gca()  # 获取当前轴
ax.spines['top'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)


plt.xlabel('Iterations')
plt.ylabel('MAE')
plt.xlim(0, 160)  # 注意修改以匹配数据点数量
plt.ylim(1.1, 2.2)
legend = plt.legend(loc='upper right', ncol=2, columnspacing=0.8)
legend.get_frame().set_linewidth(2)  # 加粗图例边框
plt.grid(True, linestyle='-', linewidth=2)  # 使网格线更粗
plt.tight_layout()
# 完成图表的绘制后，保存为高画质PDF
plt.savefig('mae_5s.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()


