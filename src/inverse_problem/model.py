import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 1. Parquet Dataset Loader
class TransferFunctionDataset(Dataset):
    def __init__(self, parquet_filepath):
        """
        Expects a parquet file where rows are different simulation runs.
        Columns should contain the flattened amplitude/phase arrays for 10 nodes,
        plus the 81 target L and C values.
        """
        # Load data efficiently using pandas
        self.data = pd.read_parquet(parquet_filepath)

        # NOTE: You MUST scale your targets before training.
        # e.g., multiply L by 1e6 (to micro) and C by 1e9 (to nano) so targets are ~O(1)
        self.targets = self.data[
            [f"L_{i}" for i in range(1, 41)] + [f"C_{i}" for i in range(41)]
        ].values.astype(np.float32)

        # Extract features and reshape to (Channels, Sequence_Length)
        # Assuming your dataframe stores lists/arrays of frequency bins for each channel
        feature_cols = [f"Node_{n}_Amp" for n in range(10)] + [
            f"Node_{n}_Phase" for n in range(10)
        ]

        # Stack into shape: (Num_Samples, 20_Channels, Num_Freq_Bins)
        self.features = np.stack(
            [np.vstack(self.data[col].values) for col in feature_cols], axis=1
        ).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])


# 2. 1D CNN Architecture
class TransferFunctionCNN(nn.Module):
    def __init__(self, num_nodes=10, num_outputs=81):
        super(TransferFunctionCNN, self).__init__()

        # 10 nodes * 2 (Amplitude & Phase) = 20 input channels
        in_channels = num_nodes * 2

        # Feature Extractor: 1D Convolutions
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            # Block 2
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
            # Block 3
            nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # Adaptive pool ensures a fixed output size (e.g., length 8) regardless of input frequency bins
            nn.AdaptiveAvgPool1d(8),
        )

        # Regressor Head: Fully Connected Layers
        self.regressor = nn.Sequential(
            nn.Linear(256 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),  # Essential to prevent overfitting to simulations
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs),  # 81 values (40 L + 41 C)
        )

    def forward(self, x):
        # x shape: (Batch, 20, Num_Freq_Bins)
        x = self.feature_extractor(x)

        # Flatten for the dense layers: (Batch, 256 * 8)
        x = torch.flatten(x, 1)

        x = self.regressor(x)
        return x


# --- Example Initialization ---
# model = TransferFunctionCNN()
# print(model)
