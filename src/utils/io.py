import yaml
import pandas as pd
import os
import numpy as np
import glob
from pathlib import Path


def load_config(file_path):
    """
    Loads data from a ".yaml" file into a dictionary.

    Inputs:
        file_path (str): The path to the ".yaml" file.

    Outputs:
        dict: A dictionary containing the loaded data.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def get_processed_filename(raw_filename):
    """
    Converts a raw filename (e.g., 'AMPPUL00.CSV') to its processed parquet filename
    (e.g., 'AMPPUL00.parquet').

    This function can handle a single filename string or a NumPy array of filenames.

    Inputs:
        raw_filename (str or np.ndarray): The original filename(s), potentially with a .csv extension.

    Outputs:
        str or np.ndarray: The processed filename(s) with a .parquet extension.
                           Returns a string if input was a string, and a NumPy array if input was an array.
    """
    if isinstance(raw_filename, np.ndarray):
        # Apply the transformation to each element in the array
        return np.vectorize(lambda x: Path(x).stem + ".parquet")(raw_filename)
    else:
        return Path(raw_filename).stem + ".parquet"


def save_parquet(data, output_dir, prefix):
    """
    Saves a dictionary of data into a parquet file. Takes into account previous simulations with the same prefix
    to avoid overwriting.

    Inputs:
        data (dict): The dictionary to be saved.
        output_dir (str): The directory where the parquet file will be saved.
        prefix (str): The prefix of the parquet files (e.g., 'results_', 'targets_', 'axes_').

    Outputs:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    existing_indices = []
    for f in os.listdir(output_dir):
        if f.startswith(prefix) and f.endswith(".parquet"):
            existing_indices.append(int(f[len(prefix) : -8]))

    next_idx = max(existing_indices) + 1 if existing_indices else 1

    df_freq = pd.DataFrame(data)
    df_freq.to_parquet(
        os.path.join(output_dir, prefix + f"{next_idx}.parquet"), engine="pyarrow"
    )


def load_parquet_data(data_dir, prefix):
    """
    Loads all chunked parquet files matching a specific prefix from a directory
    into a single unified Pandas DataFrame.

    Inputs:
        data_dir (str): The directory containing the parquet files.
        prefix (str): The prefix of the files to load (e.g., 'results_', 'targets_', 'axes_').

    Outputs:
        pd.DataFrame: A single DataFrame containing all concatenated chunks.
    """
    if not isinstance(prefix, str):
        raise TypeError("Prefix must be a string.")

    # Search for all files in the directory that match the prefix
    search_pattern = os.path.join(data_dir, f"{prefix}*.parquet")
    files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(
            f"No files matching '{prefix}*.parquet' found in {data_dir}"
        )

    # Pandas automatically reads the list of files and concatenates them sequentially
    df = pd.read_parquet(files, engine="pyarrow")

    return df
