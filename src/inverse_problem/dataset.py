import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class TransmissionLineDataset(Dataset):
    def __init__(self, data_dir, num_samples, preload_to_ram=True):
        """
        Loads the exported parquet files.
        Preloading to RAM is highly recommended for 10k files to avoid disk I/O bottlenecks.
        """
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.preload = preload_to_ram

        self.x_data = []
        self.y_data = []

        if self.preload:
            print(
                f"Preloading {num_samples} simulations into RAM. This may take a moment..."
            )
            for i in range(1, num_samples + 1):
                x, y = self._load_single_simulation(i)
                self.x_data.append(x)
                self.y_data.append(y)
            print("Preloading complete.")

    def _load_single_simulation(self, file_idx):
        freq_path = os.path.join(self.data_dir, f"freq_data_{file_idx}.parquet")
        target_path = os.path.join(self.data_dir, f"targets_{file_idx}.parquet")

        df_freq = pd.read_parquet(freq_path)
        df_targets = pd.read_parquet(target_path)

        # Assuming df_freq columns are the 20 channels and rows are frequency bins.
        # PyTorch Conv1d expects shape: (Channels, Sequence_Length)
        # Therefore, we extract values and transpose (.T)
        freq_tensor = torch.tensor(df_freq.values.T, dtype=torch.float32)

        # Extract normalized targets (C_norm and L_norm, 41 each) and concatenate to 82 values
        c_norm = df_targets["C_norm"].values
        l_norm = df_targets["L_norm"].values
        target_tensor = torch.tensor(
            np.concatenate([c_norm, l_norm]), dtype=torch.float32
        )

        return freq_tensor, target_tensor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.preload:
            return self.x_data[idx], self.y_data[idx]
        else:
            # 1-based indexing for filenames
            return self._load_single_simulation(idx + 1)
