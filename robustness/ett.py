import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import copy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CSV_FILE_PATH = '../../ICL4DT/data/time_series_datasets/ETTm1.csv'  
TARGET_COLUMN = 'OT'
INPUT_WINDOW_SIZE = 48
OUTPUT_WINDOW_SIZE = 48

TRAIN_AND_VALID_INPUT_NOISE_STD = 0.1
TEST_INPUT_NOISE_STD = 0       

NUM_FEATURES = 1
EMBED_DIM = 32
NUM_HEADS = 2
FF_DIM = 32
NUM_TRANSFORMER_BLOCKS = 2
MLP_UNITS = [64]
DROPOUT_RATE = 0.1
MLP_DROPOUT_RATE = 0.1
EPOCHS = 1000
BATCH_SIZE = 32 
LEARNING_RATE = 1e-3
PATIENCE_EARLY_STOPPING = 100

def add_gaussian_noise_to_inputs(data_array, std_dev):
    if std_dev > 0:
        noise = np.random.normal(0, std_dev, data_array.shape).astype(np.float32)
        return data_array + noise
    return data_array

def create_windows(data, input_window_size, output_window_size, stride=1):
    inputs, outputs = [], []
    data_len = len(data)
    for i in range(0, data_len - input_window_size - output_window_size + 1, stride):
        inputs.append(data[i:(i + input_window_size)])
        outputs.append(data[(i + input_window_size):(i + input_window_size + output_window_size)])
    return np.array(inputs), np.array(outputs)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(ff_dim, embed_dim))
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    def forward(self, x):
        attn_output, _ = self.att(x, x, x, need_weights=False)
        x = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output))
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_seq_len, num_features, output_seq_len, embed_dim, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout_rate=0.1, mlp_dropout_rate=0.1):
        super().__init__()
        self.input_projection = nn.Linear(num_features, embed_dim) if num_features != embed_dim else nn.Identity()
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=input_seq_len + 200)
        self.transformer_encoder_blocks = nn.Sequential(*[TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_transformer_blocks)])
        self.flatten = nn.Flatten()
        mlp_layers = []
        current_dim = embed_dim * input_seq_len
        for units in mlp_units:
            mlp_layers.extend([nn.Linear(current_dim, units), nn.ReLU(), nn.Dropout(mlp_dropout_rate)])
            current_dim = units
        mlp_layers.append(nn.Linear(current_dim, output_seq_len * num_features))
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_reshape_layer = nn.Unflatten(dim=1, unflattened_size=(output_seq_len, num_features))
    def forward(self, src):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder_blocks(src)
        out = self.flatten(memory)
        out = self.mlp(out)
        out = self.output_reshape_layer(out)
        return out

# --- Data Loading and Preprocessing ---
print("Loading and preprocessing data...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
    series_data = df[TARGET_COLUMN].dropna().values.astype(np.float32)
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILE_PATH}")
    exit()
except KeyError:
    print(f"Error: Column '{TARGET_COLUMN}' not found in the CSV.")
    exit()

if len(series_data) < (INPUT_WINDOW_SIZE + OUTPUT_WINDOW_SIZE) * 3: # Min data for one sample in each set
    print(f"Error: Not enough data after dropping NaNs ({len(series_data)} points) to create train/val/test splits and windows.")
    exit()

series_data = series_data.reshape(-1, 1)

data_len = len(series_data)
train_val_split_idx = int(data_len * 0.8)
val_test_split_idx = int(data_len * 0.9)

train_series = series_data[:train_val_split_idx]
val_series = series_data[train_val_split_idx:val_test_split_idx]
test_series = series_data[val_test_split_idx:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_series_scaled = scaler.fit_transform(train_series)
val_series_scaled = scaler.transform(val_series)
test_series_scaled = scaler.transform(test_series)

print(f"Data loaded. Train shape: {train_series_scaled.shape}, Val shape: {val_series_scaled.shape}, Test shape: {test_series_scaled.shape}")

X_train_norm_clean_np, y_train_norm_clean_np = create_windows(train_series_scaled, INPUT_WINDOW_SIZE, OUTPUT_WINDOW_SIZE)
X_val_norm_clean_np, y_val_norm_clean_np = create_windows(val_series_scaled, INPUT_WINDOW_SIZE, OUTPUT_WINDOW_SIZE)
X_test_norm_clean_np, y_test_norm_clean_np = create_windows(test_series_scaled, INPUT_WINDOW_SIZE, OUTPUT_WINDOW_SIZE)

if len(X_train_norm_clean_np) == 0 or len(X_val_norm_clean_np) == 0 or len(X_test_norm_clean_np) == 0:
    print("Error: Not enough data in one or more sets to create any windows.")
    print(f"Available for training after windowing: {len(X_train_norm_clean_np)}")
    print(f"Available for validation after windowing: {len(X_val_norm_clean_np)}")
    print(f"Available for testing after windowing: {len(X_test_norm_clean_np)}")
    exit()


print(f"Windowing complete. Shapes: X_train: {X_train_norm_clean_np.shape}, y_train: {y_train_norm_clean_np.shape}")
print(f"X_val: {X_val_norm_clean_np.shape}, y_val: {y_val_norm_clean_np.shape}")
print(f"X_test: {X_test_norm_clean_np.shape}, y_test: {y_test_norm_clean_np.shape}")

print(f"Adding Gaussian noise (std={TRAIN_AND_VALID_INPUT_NOISE_STD}) to training and validation input windows...")
X_train_norm_noisy_np = add_gaussian_noise_to_inputs(X_train_norm_clean_np, TRAIN_AND_VALID_INPUT_NOISE_STD)
X_val_norm_noisy_np = add_gaussian_noise_to_inputs(X_val_norm_clean_np, TRAIN_AND_VALID_INPUT_NOISE_STD)

X_train = torch.from_numpy(X_train_norm_noisy_np).float().to(device)
y_train = torch.from_numpy(y_train_norm_clean_np).float().to(device) 
X_val = torch.from_numpy(X_val_norm_noisy_np).float().to(device)
y_val = torch.from_numpy(y_val_norm_clean_np).float().to(device)  

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples available: {len(X_test_norm_clean_np)}")

print("Building PyTorch Transformer model...")
model = TimeSeriesTransformer(
    input_seq_len=INPUT_WINDOW_SIZE, num_features=NUM_FEATURES, output_seq_len=OUTPUT_WINDOW_SIZE,
    embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM,
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS, mlp_units=MLP_UNITS,
    dropout_rate=DROPOUT_RATE, mlp_dropout_rate=MLP_DROPOUT_RATE
).to(device)
print(f"Model Summary: Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Training model...")
train_losses, val_losses = [], []
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs_model = model(inputs)
        loss = criterion(outputs_model, targets)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs.size(0)
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_model = model(inputs)
            loss = criterion(outputs_model, targets)
            running_val_loss += loss.item() * inputs.size(0)
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        best_model_state = copy.deepcopy(model.state_dict())
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= PATIENCE_EARLY_STOPPING:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

if best_model_state:
    print("Loading best model weights for final predictions.")
    model.load_state_dict(best_model_state)
else:
    print("Warning: Early stopping did not trigger; using model from last epoch.")

if len(X_test_norm_clean_np) > 0:
    print(f"Evaluating on Test Set... Adding noise (std={TEST_INPUT_NOISE_STD}) to test inputs.")
    X_test_norm_noisy_np = add_gaussian_noise_to_inputs(X_test_norm_clean_np, TEST_INPUT_NOISE_STD)
    
    X_test_tensor = torch.from_numpy(X_test_norm_noisy_np).float().to(device)
    y_test_tensor = torch.from_numpy(y_test_norm_clean_np).float().to(device) 
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval() 
    test_loss_total = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_model = model(inputs)
            loss = criterion(outputs_model, targets)
            test_loss_total += loss.item() * inputs.size(0)
    
    avg_test_mse = test_loss_total / len(test_loader.dataset)
    print(f"Test Set MSE (on normalized data, inputs noisy std={TEST_INPUT_NOISE_STD}): {avg_test_mse:.6f}")
else:
    print("No test data to evaluate.")

# --- Prediction and Plotting ---
print("Generating and plotting predictions...")
if len(X_test_norm_clean_np) == 0:
    print("No test samples to plot.")
else:
    test_sample_idx = 1
    test_input_sample_norm_clean_np = X_test_norm_clean_np[test_sample_idx : test_sample_idx + 1]
    true_output_sample_norm_clean_np = y_test_norm_clean_np[test_sample_idx]

    print(f"Adding Gaussian noise (std={TEST_INPUT_NOISE_STD}) to the selected normalized test input window...")
    test_input_sample_norm_noisy_np = add_gaussian_noise_to_inputs(test_input_sample_norm_clean_np.copy(), TEST_INPUT_NOISE_STD)
    
    test_input_tensor_norm_noisy = torch.from_numpy(test_input_sample_norm_noisy_np).float().to(device)

    predictions_list_norm = []
    num_predictions_to_plot = 5
    model.train()
    with torch.no_grad():
        for _ in range(num_predictions_to_plot):
            prediction_tensor_norm = model(test_input_tensor_norm_noisy)
            predictions_list_norm.append(prediction_tensor_norm.cpu().numpy().squeeze())
    model.eval()

    test_input_plot_orig_scale = scaler.inverse_transform(test_input_sample_norm_clean_np.reshape(INPUT_WINDOW_SIZE, NUM_FEATURES))
    noisy_test_input_plot_orig_scale = scaler.inverse_transform(test_input_sample_norm_noisy_np.reshape(INPUT_WINDOW_SIZE, NUM_FEATURES))
    true_output_plot_orig_scale = scaler.inverse_transform(true_output_sample_norm_clean_np.reshape(OUTPUT_WINDOW_SIZE, NUM_FEATURES))
    
    predictions_list_orig_scale = []
    for pred_norm in predictions_list_norm:
        pred_orig_scale = scaler.inverse_transform(pred_norm.reshape(OUTPUT_WINDOW_SIZE, NUM_FEATURES))
        predictions_list_orig_scale.append(pred_orig_scale.squeeze())


    plt.figure(figsize=(15, 8))
    time_axis_input = np.arange(INPUT_WINDOW_SIZE)
    time_axis_output = np.arange(INPUT_WINDOW_SIZE, INPUT_WINDOW_SIZE + OUTPUT_WINDOW_SIZE)

    plt.plot(time_axis_input, test_input_plot_orig_scale.squeeze(), 'c-', alpha=0.7, label='Original Clean Test Input (Original Scale)')
    plt.plot(time_axis_input, noisy_test_input_plot_orig_scale.squeeze(), 'bo-', label=f'Noisy Test Input (Original Scale, Model sees norm+noise std={TEST_INPUT_NOISE_STD})')
    plt.plot(time_axis_output, true_output_plot_orig_scale.squeeze(), 'go-', label='True Output (Original Scale)')

    for i, pred_series_orig_scale in enumerate(predictions_list_orig_scale):
        plt.plot(time_axis_output, pred_series_orig_scale, linestyle='--', marker='x', alpha=0.7, label=f'Prediction Sample {i+1} (Original Scale)')
    
    plt.title(f"Transformer Forecast: Train/Val Norm-Input-Noise std={TRAIN_AND_VALID_INPUT_NOISE_STD}, Test Norm-Input-Noise std={TEST_INPUT_NOISE_STD}")
    plt.xlabel("Time Step")
    plt.ylabel(f"{TARGET_COLUMN} Value (Original Scale)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/ett_preds_train_{TRAIN_AND_VALID_INPUT_NOISE_STD}_test_{TEST_INPUT_NOISE_STD}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=f'Training Loss (Norm Inputs noisy std={TRAIN_AND_VALID_INPUT_NOISE_STD})')
    plt.plot(val_losses, label=f'Validation Loss (Norm Inputs noisy std={TRAIN_AND_VALID_INPUT_NOISE_STD})')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (on normalized data)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/ett_loss_train_{TRAIN_AND_VALID_INPUT_NOISE_STD}_test_{TEST_INPUT_NOISE_STD}.png")

print("Script finished. Plots saved.")