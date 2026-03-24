import yaml
import pandas as pd
import os
import glob


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


def save_parquet(data, output_dir, name):
    """
    Saves a dictionary of data into a parquet file. Takes into account previous simulations with the same name
    to avoid overwriting.

    Inputs:
        data (dict): The dictionary to be saved.
        output_dir (str): The directory where the parquet file will be saved.
        name (str): The base name for the parquet file.

    Outputs:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not isinstance(name, str):
        raise TypeError("Name must be a string.")

    existing_indices = []
    for f in os.listdir(output_dir):
        if f.startswith(name) and f.endswith(".parquet"):
            existing_indices.append(int(f[len(name) : -8]))

    next_idx = max(existing_indices) + 1 if existing_indices else 1

    df_freq = pd.DataFrame(data)
    df_freq.to_parquet(
        os.path.join(output_dir, name + f"{next_idx}.parquet"), engine="pyarrow"
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
