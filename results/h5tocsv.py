import h5py
import pandas as pd

# 定义HDF5文件列表
hdf5_files = ['driving10_fg234_FedAvg_test_2.h5', 'driving10_fg234_FedProx_test_2.h5','driving10_fg234_PerAvg_test_2.h5',
             'driving10_fg234_FedPer_test_2.h5', 'driving10_fg234_FedRep_test_2.h5','driving10_fg234_pFedMe_test_2.h5',
             'driving10_fg234_Ditto_test_2.h5', 'driving10_fg234_APFL_test_2.h5','driving10_fg234_FedFomo_test_2.h5',
            'driving10_fg234_FedALA_test_2.h5', 'driving10_fg234_FedAWA_test_2.h5'
              ]
names =['FedAvg','FedProx','Per-FedAvg','FedPer','FedRep','pFedMe','Ditto','APFL','FedFomo','FedALA','FedPAW']

for hdf5_file in hdf5_files:
    with h5py.File(hdf5_file, 'r') as hf:
        rs_test_mae = hf['rs_test_mae'][:]
        rs_test_rmse = hf['rs_test_rmse'][:]
        rs_train_loss = hf['rs_train_loss'][:]

    # 确定最小长度
    min_length = min(len(rs_test_mae), len(rs_test_rmse), len(rs_train_loss))

    # 裁剪数组以确保相同长度
    rs_test_mae = rs_test_mae[:min_length]
    rs_test_rmse = rs_test_rmse[:min_length]
    rs_train_loss = rs_train_loss[:min_length]

    data = {
        'mae': rs_test_mae,
        'rmse': rs_test_rmse,
        'loss': rs_train_loss
    }
    df = pd.DataFrame(data)

    csv_file_path = hdf5_file.replace('.h5', '.csv')
    df.to_csv(csv_file_path, index=False)

    print(f"Data from {hdf5_file} has been successfully exported to {csv_file_path}.")
