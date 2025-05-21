import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_AND_VALID_INPUT_NOISE_STD = 0.4
TEST_INPUT_NOISE_STD = 0.5

def generate_synthetic_data(num_total_points=20000, num_series_features=1):
    t = np.arange(0, num_total_points)
    series = np.sin(0.02 * t) + np.sin(0.05 * t) * 0.5 + np.cos(0.1 * t) * 0.3
    if num_series_features == 1:
        return series.astype(np.float32).reshape(-1, 1)
    else:
        return np.tile(series.astype(np.float32).reshape(-1, 1), (1, num_series_features))

def add_gaussian_noise_to_inputs(data_array, std_dev):
    if std_dev > 0:
        noise = np.random.normal(0, std_dev, data_array.shape).astype(np.float32)
        return data_array + noise
    return data_array

def create_windows(data, input_window_size, output_window_size, stride=1):
    inputs = []
    outputs = []
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
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=input_seq_len + 10)
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

INPUT_WINDOW_SIZE = 30
OUTPUT_WINDOW_SIZE = 30
NUM_FEATURES = 1
EMBED_DIM = 32
NUM_HEADS = 2
FF_DIM = 32
NUM_TRANSFORMER_BLOCKS = 2
MLP_UNITS = [64]
DROPOUT_RATE = 0.15
MLP_DROPOUT_RATE = 0.15
EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE_EARLY_STOPPING = 10

print("Generating synthetic data...")
series_data_clean = generate_synthetic_data(num_total_points=30000, num_series_features=NUM_FEATURES)
print("Creating clean input/output windows...")
X_np_clean, y_np_clean = create_windows(series_data_clean, INPUT_WINDOW_SIZE, OUTPUT_WINDOW_SIZE, stride=1)

train_split_idx = int(0.8 * len(X_np_clean))
val_split_idx = int(0.9 * len(X_np_clean))

X_train_np_orig = X_np_clean[:train_split_idx]
X_val_np_orig = X_np_clean[train_split_idx:val_split_idx]
X_test_np_orig = X_np_clean[val_split_idx:]

y_train_np = y_np_clean[:train_split_idx]
y_val_np = y_np_clean[train_split_idx:val_split_idx]
y_test_np = y_np_clean[val_split_idx:]

print(f"Adding Gaussian noise (std={TRAIN_AND_VALID_INPUT_NOISE_STD}) to training and validation input windows...")
X_train_np_noisy = add_gaussian_noise_to_inputs(X_train_np_orig, TRAIN_AND_VALID_INPUT_NOISE_STD)
X_val_np_noisy = add_gaussian_noise_to_inputs(X_val_np_orig, TRAIN_AND_VALID_INPUT_NOISE_STD)

X_train = torch.from_numpy(X_train_np_noisy).float().to(device)
y_train = torch.from_numpy(y_train_np).float().to(device)
X_val = torch.from_numpy(X_val_np_noisy).float().to(device)
y_val = torch.from_numpy(y_val_np).float().to(device)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(X_train)} (inputs noisy std={TRAIN_AND_VALID_INPUT_NOISE_STD})")
print(f"Validation samples: {len(X_val)} (inputs noisy std={TRAIN_AND_VALID_INPUT_NOISE_STD})")
print(f"Test samples available: {len(X_test_np_orig)}")

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
        outputs = model(inputs)
        loss = criterion(outputs, targets)
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
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * inputs.size(0)
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f} (Inputs noisy std={TRAIN_AND_VALID_INPUT_NOISE_STD})")

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

print("Generating and plotting predictions...")
test_sample_idx = 0
test_input_sample_clean_np = X_test_np_orig[test_sample_idx : test_sample_idx + 1]
print(f"Adding Gaussian noise (std={TEST_INPUT_NOISE_STD}) to the selected test input window...")
test_input_sample_noisy_np = add_gaussian_noise_to_inputs(test_input_sample_clean_np.copy(), TEST_INPUT_NOISE_STD)
true_output_sample_np = y_test_np[test_sample_idx]
test_input_tensor_noisy = torch.from_numpy(test_input_sample_noisy_np).float().to(device)

predictions_list = []
num_predictions_to_plot = 5
model.train()
with torch.no_grad():
    for _ in range(num_predictions_to_plot):
        prediction_tensor = model(test_input_tensor_noisy)
        predictions_list.append(prediction_tensor.squeeze().cpu().numpy())
model.eval()

plt.figure(figsize=(15, 8))
plt.plot(np.arange(INPUT_WINDOW_SIZE), test_input_sample_clean_np.squeeze(), 'b-', alpha=0.5, label='Original Clean Test Input')
plt.plot(np.arange(INPUT_WINDOW_SIZE), test_input_sample_noisy_np.squeeze(), 'bo-', label=f'Noisy Test Input (std={TEST_INPUT_NOISE_STD})')
output_time_axis = np.arange(INPUT_WINDOW_SIZE, INPUT_WINDOW_SIZE + OUTPUT_WINDOW_SIZE)
plt.plot(output_time_axis, true_output_sample_np.squeeze(), 'go-', label='True Output (Clean)')
for i, pred_series in enumerate(predictions_list):
    plt.plot(output_time_axis, pred_series, linestyle='--', marker='x', alpha=0.7, label=f'Prediction Sample {i+1}')
plt.title(f"Transformer: Train/Val Input Noise std={TRAIN_AND_VALID_INPUT_NOISE_STD}, Test Input Noise std={TEST_INPUT_NOISE_STD}")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots/preds_train_val_std_{TRAIN_AND_VALID_INPUT_NOISE_STD}_test_std_{TEST_INPUT_NOISE_STD}.png")
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label=f'Training Loss (Inputs noisy std={TRAIN_AND_VALID_INPUT_NOISE_STD})')
plt.plot(val_losses, label=f'Validation Loss (Inputs noisy std={TRAIN_AND_VALID_INPUT_NOISE_STD})')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.savefig(f"plots/loss_train_val_std_{TRAIN_AND_VALID_INPUT_NOISE_STD}_test_std_{TEST_INPUT_NOISE_STD}.png")
# plt.show()

print("Script finished. Plots saved.")