from datetime import time

import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader, ConcatDataset
from flcore.trainmodel.models import MultiHead_Seq2SeqLSTM, DrivingLSTM
from utils.data_utils import read_client_data, get_scaler_target

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on centralized data')
    parser.add_argument('-d', "--device", type=str, default='cuda:0', help='Device to use for training (default: cuda:0)')
    parser.add_argument('-b', "--batch_size", type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.004, help='Learning rate (default: 0.004)')
    parser.add_argument('-e', "--epochs", type=int, default=50, help='Number of epochs to train (default: 300)')
    parser.add_argument('-data',"--dataset", type=str, default='driving', help='Dataset to use (default: driving)')
    parser.add_argument('-t', "--trials", type=int, default=5, help='Number of independent trials to run (default: 5)')
    parser.add_argument('-m', "--model", type=str, default="sslstm")

    return parser.parse_args()

args = parse_args()

# 确保设备选择是有效的
if not torch.cuda.is_available():
    args.device = 'cpu'

# 加载数据集
def load_all_data():
    all_train_data = []
    all_test_data = []
    for client_id in range(10):  # 假设有10个客户端
        train_data = read_client_data(args.dataset, client_id, is_train=True)
        test_data = read_client_data(args.dataset, client_id, is_train=False)
        all_train_data.append(train_data)
        all_test_data.append(test_data)
    return ConcatDataset(all_train_data), ConcatDataset(all_test_data)


# 定义模型
if args.model == "sslstm":
    if "driving" == args.dataset:
        model = MultiHead_Seq2SeqLSTM(input_dim=12, hidden_dim=128, decoder_output_dim=1, num_layers=2,
                                           future_seq_length=5, dropout=0.1).to(args.device)
    elif "driving_fg2" == args.dataset or "driving_fg3" == args.dataset or "driving_fg4" == args.dataset:
        model = MultiHead_Seq2SeqLSTM(input_dim=15, hidden_dim=128, decoder_output_dim=1, num_layers=2,
                                           future_seq_length=5, dropout=0.1).to(args.device)
    elif "driving_fg24" == args.dataset or "driving_fg34" == args.dataset:
        model = MultiHead_Seq2SeqLSTM(input_dim=18, hidden_dim=128, decoder_output_dim=1, num_layers=2,
                                           future_seq_length=5, dropout=0.1).to(args.device)
    elif "driving_fg234" == args.dataset:
        model = MultiHead_Seq2SeqLSTM(input_dim=21, hidden_dim=128, decoder_output_dim=1, num_layers=2,
                                           future_seq_length=5, dropout=0.1).to(args.device)
    elif "driving10" == args.dataset:
        model = MultiHead_Seq2SeqLSTM(input_dim=17, hidden_dim=128, decoder_output_dim=1, num_layers=3,
                                           future_seq_length=10, dropout=0.2).to(args.device)
    elif "driving10_fg2" == args.dataset or "driving10_fg3" == args.dataset or "driving10_fg4" == args.dataset:
        model = MultiHead_Seq2SeqLSTM(input_dim=20, hidden_dim=128, decoder_output_dim=1, num_layers=3,
                                           future_seq_length=10, dropout=0.2).to(args.device)
    elif "driving10_fg24" == args.dataset or "driving10_fg34" == args.dataset:
        model = MultiHead_Seq2SeqLSTM(input_dim=23, hidden_dim=128, decoder_output_dim=1, num_layers=3,
                                           future_seq_length=10, dropout=0.2).to(args.device)
    elif "driving10_fg234" == args.dataset:
        model = MultiHead_Seq2SeqLSTM(input_dim=26, hidden_dim=128, decoder_output_dim=1, num_layers=3,
                                           future_seq_length=10, dropout=0.2).to(args.device)

elif args.model == "lstm":
    if "driving" ==  args.dataset:
        model = DrivingLSTM(input_dim=12, hidden_dim=128, num_layers=2, bidirectional=False,
                                         dropout=0.1).to(args.device)
    elif "driving_fg24" == args.dataset or "driving_fg34" == args.dataset or "driving_fg34_jr" == args.dataset:
        model = DrivingLSTM(input_dim=18, hidden_dim=128, num_layers=2, bidirectional=False,
                                         dropout=0.1).to(args.device)
    elif "driving10" == args.dataset:
        model = DrivingLSTM(input_dim=17, hidden_dim=128, num_layers=2, bidirectional=False,
                                         dropout=0.1).to(args.device)
    elif "driving10_fg24" == args.dataset or "driving10_fg34" == args.dataset or "driving10_fg34_jr" == args.dataset:
        model = DrivingLSTM(input_dim=23, hidden_dim=128, num_layers=2, bidirectional=False,
                                         dropout=0.1).to(args.device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# 训练模型
def train_model(train_data, test_data):
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    mae_history, rmse_history = [], []  # 用于跟踪每个epoch的MAE和RMSE

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss / len(train_loader)}")

        # 每一轮训练后进行一次测试
        print(f"Testing after epoch {epoch + 1}:")
        mae, rmse = test_model(test_data)
        mae_history.append(mae)
        rmse_history.append(rmse)
        print(f"MAE: {mae}, RMSE: {rmse}")
        model.train()

    lowest_mae = min(mae_history)
    lowest_rmse = min(rmse_history)
    print(f"Training completed. Lowest MAE: {lowest_mae}, Lowest RMSE: {lowest_rmse}")

    return model, lowest_mae, lowest_rmse



def test_model(test_data):
    model.eval()
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    total_mae = total_rmse = total_samples = 0
    scaler = get_scaler_target(args.dataset)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(args.device), y.to(args.device)
            output = model(x)
            print(y.shape)
            print(output.shape)
            time(10)
            output_reshaped = output.view(-1, 1).cpu().numpy()
            y_reshaped = y.view(-1, 1).cpu().numpy()
            output_inversed = scaler.inverse_transform(output_reshaped)
            y_inversed = scaler.inverse_transform(y_reshaped)
            output_inversed = output_inversed.reshape(-1, output.size(1), 1)
            y_inversed = y_inversed.reshape(-1, y.size(1), 1)

            for i in range(output_inversed.shape[0]):
                sequence_rmse = np.sqrt(np.mean((output_inversed[i] - y_inversed[i]) ** 2))
                sequence_mae = np.mean(np.abs(output_inversed[i] - y_inversed[i]))
                total_rmse += sequence_rmse
                total_mae += sequence_mae
                total_samples += 1
    mae = total_mae / total_samples
    rmse = total_rmse / total_samples
    return mae, rmse


if __name__ == "__main__":
    args = parse_args()

    all_lowest_maes = []
    all_lowest_rmses = []

    for trial in range(args.trials):
        print(f"Trial {trial + 1}/{args.trials}...")
        train_data, test_data = load_all_data()
        train_model(train_data, test_data)

        # 收集每次试验的最低MAE和RMSE
        _, lowest_mae, lowest_rmse = train_model(train_data, test_data)
        all_lowest_maes.append(lowest_mae)
        all_lowest_rmses.append(lowest_rmse)

    # 打印每次试验的最低MAE和RMSE的平均值和标准差
    print("=" * 50)
    print("Algorithm: Cloud")
    print("Epochs: {}".format(args.epochs))
    print("Dataset: {}".format(args.dataset))
    print("=" * 50)
    print("std for lowest mae:", np.std(all_lowest_maes))
    print("mean for lowest mae:", np.mean(all_lowest_maes))
    print("std for lowest rmse:", np.std(all_lowest_rmses))
    print("mean for lowest rmse:", np.mean(all_lowest_rmses))
