import torch
import torch.nn as nn
import torch.nn.functional as F


class TransferFunction2DCNN(nn.Module):
    def __init__(self, num_outputs=82):
        super(TransferFunction2DCNN, self).__init__()

        # Input shape will be (Batch, 2, 10, 160)
        # Channel 0: Magnitude, Channel 1: Phase
        # Height: 10 Spatial Nodes, Width: 160 Frequency Bins

        self.feature_extractor = nn.Sequential(
            # Layer 1: Look at 3 adjacent nodes and 5 frequency bins
            nn.Conv2d(
                in_channels=2, out_channels=32, kernel_size=(3, 5), padding=(1, 2)
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # Pool only across frequencies to preserve spatial resolution early on
            nn.MaxPool2d(kernel_size=(1, 2)),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Height becomes 5, Width becomes 40
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((5, 10)),  # Compress frequency dimension, keep spatial
        )

        self.regressor = nn.Sequential(
            nn.Linear(128 * 5 * 10, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs),  # 41 C_norm + 41 L_norm
        )

    def forward(self, x):
        # x is originally (Batch, 20, 160) from the DataLoader
        # We assume alternating Mag/Phase: [Mag_4, Phase_4, Mag_8, Phase_8, ...]
        batch_size = x.size(0)

        # Reshape to (Batch, 10 nodes, 2 channels, 160 freq)
        x = x.view(batch_size, 10, 2, -1)

        # Permute to (Batch, 2 channels, 10 nodes, 160 freq) for Conv2d
        x = x.permute(0, 2, 1, 3)

        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


class TransferFunction1DCNN(nn.Module):
    def __init__(self, in_channels=2, num_outputs=82):
        super(TransferFunction1DCNN, self).__init__()

        # Input shape: (Batch, 2, 160)
        # Channel 0: Magnitude of Node 40, Channel 1: Phase of Node 40
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
            nn.Linear(512, num_outputs),  # 41 C_norm + 41 L_norm
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x
