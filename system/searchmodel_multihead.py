import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
from flcore.trainmodel.models import MultiHead_Seq2SeqLSTM
import json

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

def objective(trial):
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers',2, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.2)  # Search for dropout rate between 0 and 0.5
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-2)
    model = MultiHead_Seq2SeqLSTM(input_dim=18, hidden_dim=hidden_dim, decoder_output_dim=1, num_layers=num_layers, future_seq_length=5, dropout=dropout, num_heads=8)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_loader = DataLoader(read_client_data("driving_fg24", 0, True), batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(read_client_data("driving_fg24", 0, False), batch_size=64, shuffle=False, drop_last=False)

    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            val_loss += criterion(output, y).item()

    return val_loss / len(val_loader)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

best_trial_params = study.best_trial.params
with open('best_trial_params_multihead_seq2seq_lstm_5.json', 'w') as outfile:
    json.dump(best_trial_params, outfile, indent=4)
print('Best trial:', study.best_trial.params)
print("Saved best trial parameters for MultiHead_Seq2SeqLSTM LSTM to best_trial_params_seq2seq_lstm_5.json")



'''
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
import json

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

class CustomLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, layer_configs):
        super(CustomLSTM, self).__init__()
        layers = []
        for i, config in enumerate(layer_configs):
            layer_input_dim = input_dim if i == 0 else layer_configs[i - 1]['hidden_dim']
            layers.append(nn.LSTM(input_size=layer_input_dim,
                                  hidden_size=config['hidden_dim'],
                                  num_layers=1,  # Each LSTM() call creates 1 layer here
                                  bidirectional=False,
                                  dropout=config['dropout'],
                                  batch_first=True))
        self.lstm_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(layer_configs[-1]['hidden_dim'], output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = self.fc(x)
        x = self.relu(x)


def objective(trial):
    num_layers = trial.suggest_int('num_layers', 2, 4)
    layer_configs = []
    for i in range(num_layers):
        hidden_dim = trial.suggest_categorical(f'layer_{i}_hidden_dim', [128, 256, 512])
        dropout = trial.suggest_float(f'layer_{i}_dropout', 0.0,
                                      0.3) if i < num_layers - 1 else 0  # Dropout not applied on the last layer
        layer_configs.append({'hidden_dim': hidden_dim, 'dropout': dropout})

    model = CustomLSTM(input_dim=12, output_dim=1, layer_configs=layer_configs)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_loguniform('learning_rate', 1e-3, 1e-1))
    criterion = nn.MSELoss()

    train_loader = DataLoader(read_client_data("driving", 0, True),
                              batch_size=trial.suggest_categorical('batch_size', [32, 64, 128, 256]), shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(read_client_data("driving", 0, False),
                            batch_size=trial.suggest_categorical('batch_size', [32, 64, 128, 256]), shuffle=False,
                            drop_last=False)

    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            val_loss += criterion(output, y).item()

    return val_loss / len(val_loader)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

print('Best trial:', study.best_trial.params)

best_trial_params = study.best_trial.params
with open('best_trial_params_custom_lstm.json', 'w') as outfile:
    json.dump(best_trial_params, outfile, indent=4)

print("Saved best trial parameters for custom LSTM to best_trial_params_custom_lstm.json")
'''