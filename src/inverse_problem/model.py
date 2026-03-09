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


class ObservabilityWeightedMSE(nn.Module):
    def __init__(self, noise_floor_threshold=1e-4, priority_factor=5.0):
        """
        noise_floor_threshold: The magnitude level where signal is considered lost.
        priority_factor: The multiplier for components adjacent to measured nodes.
        """
        super(ObservabilityWeightedMSE, self).__init__()
        self.threshold = noise_floor_threshold
        self.priority_factor = priority_factor

        # Build the static Spatial Priority Mask once during initialization
        # Length 82: Indices 0-40 are Capacitors, Indices 41-81 are Inductors
        self.spatial_mask = torch.ones(82)

        # 0-indexed measured nodes based on your V_n/V_0
        measured_nodes = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

        # 1. Highly prioritize the Gateway Stage (C_0 and L_0)
        self.spatial_mask[0] = self.priority_factor  # C_0
        self.spatial_mask[41] = (
            self.priority_factor
        )  # L_0 (Index 41 in concatenated array)

        # 2. Prioritize components adjacent to measurement probes
        for n in measured_nodes:
            # Prioritize C_n
            self.spatial_mask[n] = self.priority_factor

            # Prioritize L_{n-1}
            self.spatial_mask[41 + (n - 1)] = self.priority_factor

            # Prioritize L_n (Ensure we don't prioritize the padding at index 40)
            if n < 40:
                self.spatial_mask[41 + n] = self.priority_factor

    def forward(self, predictions, targets, inputs):
        batch_size = predictions.size(0)
        device = predictions.device

        # 1. Calculate raw MSE per component
        raw_mse = F.mse_loss(predictions, targets, reduction="none")

        # 2. Extract magnitudes to determine dynamic observability
        inputs_reshaped = inputs.view(batch_size, 10, 2, -1).permute(0, 2, 1, 3)
        magnitudes = inputs_reshaped[:, 0, :, :]
        node_signal_strength = torch.max(magnitudes, dim=2)[0]

        # 3. Create dynamic observability mask (Length 41)
        dynamic_weights = torch.ones((batch_size, 41), device=device)

        for b in range(batch_size):
            for node_idx in range(10):
                if node_signal_strength[b, node_idx] < self.threshold:
                    stage_cutoff = (node_idx + 1) * 4
                    dynamic_weights[b, stage_cutoff:] *= 0.1
                    break

        # Concatenate dynamic weights for both C and L -> Shape: (Batch, 82)
        full_dynamic_weights = torch.cat([dynamic_weights, dynamic_weights], dim=1)

        # 4. Merge Dynamic Observability with Static Spatial Priority
        # Move static mask to correct device dynamically
        final_weights = full_dynamic_weights * self.spatial_mask.to(device)

        # Apply weights and return mean loss
        weighted_loss = raw_mse * final_weights
        return weighted_loss.mean()
