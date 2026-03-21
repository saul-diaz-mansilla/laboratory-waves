import torch
import numpy as np
from torch.utils.data import Dataset
import src.utils.io as io


class TransmissionLineDataset(Dataset):
    def __init__(self, data_dir, preload_to_ram=True):
        """
        Loads the chunked parquet files using the unified io loader.
        Preloading to RAM converts the entire Pandas DataFrame into stacked
        PyTorch tensors instantly for maximum GPU throughput.
        """
        self.data_dir = data_dir
        self.preload = preload_to_ram

        # Load the unified dataframes
        print(f"Reading chunked Parquet files from {data_dir}...")
        self.df_results = io.load_parquet_data(data_dir, prefix="results_")
        self.df_targets = io.load_parquet_data(data_dir, prefix="targets_")

        # Ensure data integrity across both files
        assert len(self.df_results) == len(self.df_targets), (
            "Mismatch in number of simulations between results and targets."
        )
        self.num_samples = len(self.df_results)

        # Drop frequency axis if it accidentally leaked into results
        self.feature_cols = [c for c in self.df_results.columns if "Frequency" not in c]
        self.df_results = self.df_results[self.feature_cols]

        if self.preload:
            print(
                f"Preloading {self.num_samples} simulations into PyTorch Tensors. This will be fast..."
            )

            # --- X DATA (Features) ---
            results_arrays = [np.stack(row) for row in self.df_results.values]
            self.x_data = torch.tensor(np.array(results_arrays), dtype=torch.float32)

            # --- Y DATA (Targets) ---
            # Extract C_norm and L_norm columns (each cell is a list of length N)
            c_norm_arrays = np.stack(self.df_targets["C_norm"].values)
            l_norm_arrays = np.stack(self.df_targets["L_norm"].values)

            # Concatenate along the node dimension to create (num_samples, 2*N)
            targets_arrays = np.concatenate([c_norm_arrays, l_norm_arrays], axis=1)
            self.y_data = torch.tensor(targets_arrays, dtype=torch.float32)

            print(
                f"Preloading complete. X shape: {self.x_data.shape}, Y shape: {self.y_data.shape}"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.preload:
            return self.x_data[idx], self.y_data[idx]
        else:
            # Lazy loading directly from the DataFrame rows
            row_res = self.df_results.iloc[idx]
            x_tensor = torch.tensor(np.stack(row_res.values), dtype=torch.float32)

            c_norm = self.df_targets["C_norm"].iloc[idx]
            l_norm = self.df_targets["L_norm"].iloc[idx]
            y_tensor = torch.tensor(
                np.concatenate([c_norm, l_norm]), dtype=torch.float32
            )

            return x_tensor, y_tensor
