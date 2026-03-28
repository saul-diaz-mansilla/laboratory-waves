import os
import sys
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inverse_problem.dataset import TransmissionLineDataset
from src.inverse_problem.models import build_model_from_config
from src.inverse_problem.losses import ObservabilityWeightedMSE
import src.utils.io as io


def _load_merged_nn_config(master_config_path):
    master = io.load_config(master_config_path)
    sim_path = master["paths"]["simulation_config"]
    sim = io.load_config(sim_path)
    if "neural_network" not in sim:
        raise KeyError(
            f"Missing neural_network section in simulation config: {sim_path}"
        )
    return master, sim["neural_network"]


def main():
    parser = argparse.ArgumentParser(description="Train inverse-problem CNN models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the master experimental configuration file. Located in configs/experiment",
    )
    args = parser.parse_args()

    master_cfg, nn_cfg = _load_merged_nn_config(args.config)
    train_cfg = nn_cfg["training"]
    loss_cfg = nn_cfg["loss"]
    arch_cfg = nn_cfg["architecture"]

    sim_path = master_cfg["paths"]["simulation_config"]
    sim_exec = io.load_config(sim_path)["execution"]
    target_nodes = sim_exec["target_nodes"]
    num_nodes = len(target_nodes)

    circuit = io.load_config(master_cfg["paths"]["circuit_config"])
    n_circ = circuit["N"]["value"]
    num_outputs = 2 * n_circ

    paths = master_cfg["paths"]
    data_dir = paths["simulation_dir"]
    out_2d = paths["inverse_model_2d"]
    out_2d_mag = paths["inverse_model_2d_magnitude"]
    out_1d = paths["inverse_model_1d"]

    seed = train_cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["learning_rate"]
    train_split = train_cfg["train_split"]
    grad_clip = train_cfg["grad_clip_norm_1d"]
    sch_factor = train_cfg["lr_scheduler_1d_factor"]
    sch_patience = train_cfg["lr_scheduler_1d_patience"]

    num_freq_bins = arch_cfg["num_freq_bins"]

    full_dataset = TransmissionLineDataset(data_dir=data_dir, preload_to_ram=True)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # --- 2D (magnitude + phase) ---
    model_2d = build_model_from_config(
        "2d", num_nodes, num_freq_bins, num_outputs, arch_cfg
    ).to(device)
    criterion_2d = ObservabilityWeightedMSE(
        noise_floor_threshold=loss_cfg["noise_floor_threshold"],
        priority_factor=loss_cfg["priority_factor"],
        config_path=args.config,
        magnitude_only=False,
    )
    optimizer_2d = optim.Adam(model_2d.parameters(), lr=lr)

    def _train_epoch_2d():
        model_2d.train()
        running = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_2d.zero_grad()
            loss = criterion_2d(model_2d(inputs), targets, inputs)
            loss.backward()
            optimizer_2d.step()
            running += loss.item() * inputs.size(0)
        return running / len(train_loader.dataset)

    def _val_2d():
        model_2d.eval()
        running = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                loss = criterion_2d(model_2d(inputs), targets, inputs)
                running += loss.item() * inputs.size(0)
        return running / len(val_loader.dataset)

    print("Starting 2D (mag + phase) training...")
    for epoch in range(epochs):
        tr = _train_epoch_2d()
        va = _val_2d()
        print(f"[2D] Epoch [{epoch + 1}/{epochs}] | Train {tr:.6f} | Val {va:.6f}")

    os.makedirs(os.path.dirname(out_2d), exist_ok=True)
    torch.save(model_2d.state_dict(), out_2d)
    print(f"Saved 2D model to: {out_2d}")

    # --- 2D magnitude-only ---
    model_mag = build_model_from_config(
        "2d_magnitude", num_nodes, num_freq_bins, num_outputs, arch_cfg
    ).to(device)
    criterion_mag = ObservabilityWeightedMSE(
        noise_floor_threshold=loss_cfg["noise_floor_threshold"],
        priority_factor=loss_cfg["priority_factor"],
        config_path=args.config,
        magnitude_only=True,
    )
    optimizer_mag = optim.Adam(model_mag.parameters(), lr=lr)

    def _train_epoch_mag():
        model_mag.train()
        running = 0.0
        for inputs, targets in train_loader:
            mag = inputs[:, 0::2, :].to(device)
            targets = targets.to(device)
            optimizer_mag.zero_grad()
            loss = criterion_mag(model_mag(mag), targets, mag)
            loss.backward()
            optimizer_mag.step()
            running += loss.item() * mag.size(0)
        return running / len(train_loader.dataset)

    def _val_mag():
        model_mag.eval()
        running = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                mag = inputs[:, 0::2, :].to(device)
                targets = targets.to(device)
                loss = criterion_mag(model_mag(mag), targets, mag)
                running += loss.item() * mag.size(0)
        return running / len(val_loader.dataset)

    print("Starting 2D (magnitude-only) training...")
    for epoch in range(epochs):
        tr = _train_epoch_mag()
        va = _val_mag()
        print(f"[2D-mag] Epoch [{epoch + 1}/{epochs}] | Train {tr:.6f} | Val {va:.6f}")

    os.makedirs(os.path.dirname(out_2d_mag), exist_ok=True)
    torch.save(model_mag.state_dict(), out_2d_mag)
    print(f"Saved 2D magnitude-only model to: {out_2d_mag}")

    # --- 1D on last target node ---
    node_1d = target_nodes[-1]
    node_idx = target_nodes.index(node_1d)
    ch_start = 2 * node_idx
    ch_end = ch_start + 2

    model_1d = build_model_from_config(
        "1d", num_nodes, num_freq_bins, num_outputs, arch_cfg
    ).to(device)
    criterion_1d = nn.MSELoss()
    optimizer_1d = optim.Adam(model_1d.parameters(), lr=lr)
    scheduler_1d = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_1d, mode="min", factor=sch_factor, patience=sch_patience
    )

    def _train_epoch_1d():
        model_1d.train()
        running = 0.0
        for inputs, targets in train_loader:
            x = inputs[:, ch_start:ch_end, :].to(device)
            targets = targets.to(device)
            optimizer_1d.zero_grad()
            loss = criterion_1d(model_1d(x), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_1d.parameters(), max_norm=grad_clip)
            optimizer_1d.step()
            running += loss.item() * x.size(0)
        return running / len(train_loader.dataset)

    def _val_1d():
        model_1d.eval()
        running = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                x = inputs[:, ch_start:ch_end, :].to(device)
                targets = targets.to(device)
                loss = criterion_1d(model_1d(x), targets)
                running += loss.item() * x.size(0)
        return running / len(val_loader.dataset)

    print(f"Starting 1D training (node {node_1d})...")
    for epoch in range(epochs):
        tr = _train_epoch_1d()
        va = _val_1d()
        scheduler_1d.step(va)
        lr_cur = optimizer_1d.param_groups[0]["lr"]
        print(
            f"[1D] Epoch [{epoch + 1}/{epochs}] | Train {tr:.6f} | Val {va:.6f} | LR {lr_cur}"
        )

    os.makedirs(os.path.dirname(out_1d), exist_ok=True)
    torch.save(model_1d.state_dict(), out_1d)
    print(f"Saved 1D model to: {out_1d}")


if __name__ == "__main__":
    main()
