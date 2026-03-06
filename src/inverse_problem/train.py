import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.inverse_problem.model import TransferFunctionCNN
from src.inverse_problem.dataset import TransmissionLineDataset

# --- 1. Initialization & Data Splitting ---
data_directory = "data/inverse_problem"
total_simulations = 10000
batch_size = 64
num_epochs = 50  # Set the number of epochs here

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Load the full dataset
full_dataset = TransmissionLineDataset(
    data_dir=data_directory, num_samples=total_simulations, preload_to_ram=True
)

# Define split sizes (e.g., 80% training, 20% validation)
train_size = int(0.8 * total_simulations)
val_size = total_simulations - train_size

# Randomly split the dataset
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

# Create separate DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)  # No need to shuffle validation data

# Initialize Model, Loss, and Optimizer
model = TransferFunctionCNN(in_channels=20, num_outputs=82).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --- 2. Training and Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Set to training mode
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()  # Set to evaluation mode
    running_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


# --- 3. Multi-Epoch Execution ---
print("Starting training...")
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
    )

# --- 4. Save Final Parameters ---
save_path = "data/inverse_problem/models/transmission_line_cnn.pth"
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Training complete. Final model parameters saved to: {save_path}")
