import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Ensure the src directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the new 2D CNN and the custom physics-informed loss function
from src.inverse_problem.model import TransferFunction2DCNN, ObservabilityWeightedMSE
from src.inverse_problem.dataset import TransmissionLineDataset

# --- 1. Initialization & Data Splitting ---
data_directory = "data/inverse_problem/simulations_gaussians"
total_simulations = 10000
batch_size = 64
num_epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Load the full dataset (Target size is 82 again, nuisance parameters removed)
full_dataset = TransmissionLineDataset(
    data_dir=data_directory, num_samples=total_simulations, preload_to_ram=True
)

train_size = int(0.8 * total_simulations)
val_size = total_simulations - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# Initialize 2D Model, Custom Loss, and Optimizer
# num_outputs is 82 (41 C_norm + 41 L_norm)
model = TransferFunction2DCNN(num_outputs=82).to(device)

# Using the physics-informed loss. Adjust noise_floor_threshold based on your time-domain variance.
criterion = ObservabilityWeightedMSE(noise_floor_threshold=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --- 2. Training and Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # The custom loss now requires the raw inputs to calculate spatial observability
        loss = criterion(outputs, targets, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # The custom loss is also evaluated during validation
            loss = criterion(outputs, targets, inputs)
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
save_path = "data/inverse_problem/models/transmission_line_2dcnn.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Training complete. Final model parameters saved to: {save_path}")
