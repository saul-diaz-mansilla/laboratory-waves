import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.inverse_problem.model import TransferFunctionCNN
from src.inverse_problem.dataset import TransmissionLineDataset


# --- 1. Initialization ---
data_directory = "data/inverse_problem"
total_simulations = 10000
batch_size = 64

# Automatically utilize GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Initialize Dataset and DataLoader
dataset = TransmissionLineDataset(
    data_dir=data_directory, num_samples=total_simulations, preload_to_ram=True
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Initialize Model, Loss, and Optimizer
# Assuming 10 nodes * 2 (Amp/Phase) = 20 channels
model = TransferFunctionCNN(in_channels=20, num_outputs=82).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --- 2. Single Epoch Training Loop ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Move data to GPU/CPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            avg_loss = running_loss / 20
            print(f"Batch [{batch_idx + 1}/{len(dataloader)}] - Loss: {avg_loss:.6f}")
            running_loss = 0.0

    print("Epoch complete.")


# Run the epoch
print("Starting training epoch...")
train_one_epoch(model, dataloader, criterion, optimizer, device)

# ... (Previous training loop code) ...

print("Starting training epoch...")
train_one_epoch(model, dataloader, criterion, optimizer, device)

# --- NEW: Save the trained parameters ---
save_path = "models/transmission_line_cnn.pth"
os.makedirs("models", exist_ok=True)  # Ensure the directory exists

# Save only the state_dict (weights and biases), not the whole model class
torch.save(model.state_dict(), save_path)
print(f"Model parameters successfully saved to: {save_path}")
