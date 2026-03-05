import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.inverse_problem.model import (
    TransferFunctionCNN,
)  # Importing from your modular file

# 1. Initialize the architecture (must be the exact same structure)
model = TransferFunctionCNN(in_channels=20, num_outputs=82)

# 2. Load the trained weights from the file
load_path = "models/transmission_line_cnn.pth"
model.load_state_dict(torch.load(load_path))

# 3. CRITICAL: Set the model to evaluation mode
# This disables Dropout and freezes BatchNorm layers so they don't alter your experimental data
model.eval()

print("Model successfully loaded and ready for experimental inference.")

# ... (Code to process experimental data goes here) ...
