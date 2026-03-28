import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.io as io


def _extract_target_nodes_from_config(config_path):
    """
    Extracts `execution.target_nodes` from either:
    - the master experiment config (expects `paths.simulation_config`)
    - the simulation config directly
    """
    cfg = io.load_config(config_path)

    # Simulation config directly
    if "execution" in cfg and "target_nodes" in cfg["execution"]:
        return cfg["execution"]["target_nodes"]

    # Master experiment config -> resolve simulation config path
    if "paths" in cfg and "simulation_config" in cfg["paths"]:
        sim_path = cfg["paths"]["simulation_config"]
        sim_cfg = io.load_config(sim_path)
        return sim_cfg["execution"]["target_nodes"]

    raise ValueError(
        f"Could not extract target_nodes from config: {config_path}. "
        "Expected either simulation config with execution.target_nodes, or a master "
        "experiment config with paths.simulation_config."
    )


class ObservabilityWeightedMSE(nn.Module):
    def __init__(
        self,
        noise_floor_threshold=1e-4,
        priority_factor=5.0,
        target_nodes=None,
        config_path=None,
        magnitude_only=False,
    ):
        """
        noise_floor_threshold: The magnitude level where signal is considered lost.
        priority_factor: The multiplier for components adjacent to measured nodes.
        """
        super(ObservabilityWeightedMSE, self).__init__()
        self.threshold = noise_floor_threshold
        self.priority_factor = priority_factor
        self.measure_nodes = (
            list(target_nodes)
            if target_nodes is not None
            else (
                _extract_target_nodes_from_config(config_path)
                if config_path is not None
                else [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
            )
        )
        num_measures = len(self.measure_nodes)
        if num_measures < 1:
            raise ValueError("target_nodes/measure_nodes must contain at least one node.")

        # Build the static Spatial Priority Mask once during initialization
        # Length 82: Indices 0-40 are Capacitors, Indices 41-81 are Inductors
        self.spatial_mask = torch.ones(82)

        # 1. Highly prioritize the Gateway Stage (C_0 and L_0)
        self.spatial_mask[0] = self.priority_factor  # C_0
        self.spatial_mask[41] = (
            self.priority_factor
        )  # L_0 (Index 41 in concatenated array)

        # 2. Prioritize components adjacent to measurement probes
        for n in self.measure_nodes:
            # Prioritize C_n
            self.spatial_mask[n] = self.priority_factor

            # Prioritize L_{n-1}
            self.spatial_mask[41 + (n - 1)] = self.priority_factor

            # Prioritize L_n (Ensure we don't prioritize the padding at index 40)
            if n < 40:
                self.spatial_mask[41 + n] = self.priority_factor

        self._num_measures = num_measures
        self.magnitude_only = magnitude_only

    def forward(self, predictions, targets, inputs):
        batch_size = predictions.size(0)
        device = predictions.device

        # 1. Calculate raw MSE per component
        raw_mse = F.mse_loss(predictions, targets, reduction="none")

        # 2. Extract magnitudes to determine dynamic observability
        if self.magnitude_only:
            # (Batch, num_measures, F) — magnitude only
            magnitudes = inputs
        else:
            # (Batch, 2*num_measures, F) -> (Batch, 2, num_measures, F)
            inputs_reshaped = inputs.view(
                batch_size, self._num_measures, 2, -1
            ).permute(0, 2, 1, 3)
            magnitudes = inputs_reshaped[:, 0, :, :]
        node_signal_strength = torch.max(magnitudes, dim=2)[0]

        # 3. Create dynamic observability mask (Length 41)
        dynamic_weights = torch.ones((batch_size, 41), device=device)

        for b in range(batch_size):
            for node_idx in range(self._num_measures):
                if node_signal_strength[b, node_idx] < self.threshold:
                    # De-emphasize later stages based on the physical node index ordering
                    # used in the 2D dataset (assumed consistent with `measure_nodes`).
                    stage_cutoff = self.measure_nodes[node_idx]
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
