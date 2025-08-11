import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
from model import CNNTransformerPSSP
from data_loader import load_dataset  
scaler = torch.cuda.amp.GradScaler()


# -----------------------------
# Configs
# -----------------------------
def get_data_path(filename="cullpdb+profile_6133.npy"):
    if 'google.colab' in sys.modules:
        base_path = "/content/drive/MyDrive/PSSP Project/data"
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.normpath(os.path.join(script_dir, '..', 'data'))
    data_path = os.path.join(base_path, filename)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return data_path

DATA_PATH = get_data_path()
print(f"Using data file: {DATA_PATH}")

BATCH_SIZE = 32
MAX_EPOCHS = 3
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # autotune for fixed input sizes

# -----------------------------
# Load and preprocess data
# -----------------------------
data = load_dataset(DATA_PATH)

def get_data_labels_fixed(dataset):
    X = dataset[:, :, 35:56]  # profile input (21 dims)
    Y_raw = dataset[:, :, 22:30]  # one-hot secondary structure for Q8 (8 classes)
    mask = (np.sum(Y_raw, axis=-1) != 0)  # valid positions
    return X, Y_raw, mask

X, Y, mask = get_data_labels_fixed(data)

# Predefined split
X_train, X_test, X_val = X[:5600], X[5605:5877], X[5877:6133]
Y_train, Y_test, Y_val = Y[:5600], Y[5605:5877], Y[5877:6133]
mask_train, mask_test, mask_val = mask[:5600], mask[5605:5877], mask[5877:6133]

print("Train size:", X_train.shape[0], "Val size:", X_val.shape[0], "Test size:", X_test.shape[0])

# Convert to tensors
X_train, Y_train, mask_train = map(torch.tensor, (X_train, Y_train, mask_train))
X_val, Y_val, mask_val = map(torch.tensor, (X_val, Y_val, mask_val))
X_test, Y_test, mask_test = map(torch.tensor, (X_test, Y_test, mask_test))

# Float inputs
X_train = X_train.float()
X_val = X_val.float()
X_test = X_test.float()

# One-hot -> class index
Y_train = Y_train.argmax(dim=-1)
Y_val = Y_val.argmax(dim=-1)
Y_test = Y_test.argmax(dim=-1)

# Sanity label distribution (flattened across sequence)
print("Label counts (train):", torch.bincount(Y_train.flatten()))
print("Label counts (val):", torch.bincount(Y_val.flatten()))
print("Label counts (test):", torch.bincount(Y_test.flatten()))

# DataLoaders (tune num_workers/pin_memory if on GPU)
train_loader = DataLoader(
    TensorDataset(X_train, Y_train, mask_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True if torch.cuda.is_available() else False,
    num_workers=2
)
val_loader = DataLoader(
    TensorDataset(X_val, Y_val, mask_val),
    batch_size=BATCH_SIZE,
    pin_memory=True if torch.cuda.is_available() else False,
    num_workers=2
)
test_loader = DataLoader(
    TensorDataset(X_test, Y_test, mask_test),
    batch_size=BATCH_SIZE,
    pin_memory=True if torch.cuda.is_available() else False,
    num_workers=2
)

# -----------------------------
# Model setup
# -----------------------------
model = CNNTransformerPSSP(input_dim=21, num_classes=8).to(DEVICE)
# optional compilation speedup if using PyTorch 2.x
try:
    model = torch.compile(model)
except Exception:
    pass

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

# scheduler + early stopping
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5, verbose=True)
best_val_loss = float("inf")
patience = 2
no_improve = 0

# ensure checkpoint directory exists
os.makedirs("checkpoints", exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def evaluate(model, loader):
    model.eval()
    total_correct = 0
    total_valid = 0
    total_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch, mask in loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            mask = mask.to(DEVICE).reshape(-1).bool()  # (B*L)

            with torch.cuda.amp.autocast():
                outputs = model(X_batch)  # (B, L, C)
                B, L, C = outputs.shape
                outputs_flat = outputs.reshape(-1, C)
                labels_flat = Y_batch.reshape(-1)
                loss = criterion(outputs_flat[mask], labels_flat[mask])

            total_loss += loss.item()
            preds = outputs_flat.argmax(dim=-1)
            total_correct += (preds[mask] == labels_flat[mask]).sum().item()
            total_valid += mask.sum().item()

    accuracy = 100. * total_correct / total_valid if total_valid > 0 else 0.0
    return total_loss / len(loader), accuracy

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(MAX_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, Y_batch, mask in train_loader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)
        mask_flat = mask.to(DEVICE).reshape(-1).bool()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(X_batch)  # (B, L, C)
            B, L, C = outputs.shape
            outputs_flat = outputs.reshape(-1, C)
            labels_flat = Y_batch.reshape(-1)
            loss = criterion(outputs_flat[mask_flat], labels_flat[mask_flat])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        preds = outputs_flat.argmax(dim=-1)
        correct += (preds[mask_flat] == labels_flat[mask_flat]).sum().item()
        total += mask_flat.sum().item()

    train_acc = 100. * correct / total if total > 0 else 0.0
    val_loss, val_acc = evaluate(model, val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}: Train Loss={running_loss/len(train_loader):.4f} | "
          f"Train Acc={train_acc:.2f}% | Val Loss={val_loss:.4f} | Val Acc={val_acc:.2f}%")
    
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), "checkpoints/best.pth")
        print("   New best model saved.")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# -----------------------------
# Final evaluation
# -----------------------------
test_loss, test_acc = evaluate(model, test_loader)
print(f"\nTest Loss: {test_loss:.4f} &  Test Accuracy: {test_acc:.2f}%")

# Save final
torch.save(model.state_dict(), "checkpoints/cnn_transformer_pssp.pth")