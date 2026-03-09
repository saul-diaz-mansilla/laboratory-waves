import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.inverse_problem.model_1d import TransferFunction1DCNN
from src.inverse_problem.dataset import TransmissionLineDataset

# --- Initialization ---
data_directory = "data/inverse_problem/simulations_gaussians"
total_simulations = 10000
batch_size = 64
num_epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

full_dataset = TransmissionLineDataset(
    data_dir=data_directory, num_samples=total_simulations, preload_to_ram=True
)

train_size = int(0.8 * total_simulations)
val_size = total_simulations - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model and standard MSE Loss
model = TransferFunction1DCNN(in_channels=2, num_outputs=82).to(device)
criterion = nn.MSELoss()

# Added scheduler to fix the bouncing validation loss we discussed earlier
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)


# --- Training Loop ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        # CRITICAL: Slice the 20-channel input to only keep the last 2 channels (Node 40 Mag & Phase)
        # Shape goes from (Batch, 20, 160) -> (Batch, 2, 160)
        inputs = inputs[:, -2:, :]

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping to prevent violent spikes from unobservable components
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs[:, -2:, :]  # Slice to Node 40 only
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


# --- Execution ---
print("Starting training...")
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']}"
    )

save_path = "data/inverse_problem/models/transmission_line_1dcnn_node40.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Training complete. Final model saved to: {save_path}")
