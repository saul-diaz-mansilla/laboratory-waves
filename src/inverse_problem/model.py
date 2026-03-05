import torch
import torch.nn as nn


# 1D CNN Architecture (Updated for 82 outputs)
class TransferFunctionCNN(nn.Module):
    def __init__(self, in_channels=20, num_outputs=82):
        super(TransferFunctionCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(8),
        )

        self.regressor = nn.Sequential(
            nn.Linear(256 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs),  # Outputs 82 normalized values
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x
