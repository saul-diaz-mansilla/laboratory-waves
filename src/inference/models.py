import torch
import torch.nn as nn


def _cnn2d_backbone(
    in_channels: int,
    pool_h: int,
    pool_w: int,
    conv_channels=(32, 64, 128),
):
    """Shared feature extractor for 2D transfer-function CNNs."""
    c1, c2, c3 = conv_channels
    return nn.Sequential(
        nn.Conv2d(in_channels, c1, kernel_size=(3, 5), padding=(1, 2)),
        nn.ReLU(),
        nn.BatchNorm2d(c1),
        nn.MaxPool2d(kernel_size=(1, 2)),
        nn.Conv2d(c1, c2, kernel_size=(3, 5), padding=(1, 2)),
        nn.ReLU(),
        nn.BatchNorm2d(c2),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Conv2d(c2, c3, kernel_size=(3, 3), padding=(1, 1)),
        nn.ReLU(),
        nn.BatchNorm2d(c3),
        nn.AdaptiveAvgPool2d((pool_h, pool_w)),
    )


def _regressor_head(
    flat_dim: int,
    num_outputs: int,
    hidden1: int,
    hidden2: int,
    dropout: float,
):
    return nn.Sequential(
        nn.Linear(flat_dim, hidden1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, num_outputs),
    )


class TransferFunction2DCNN(nn.Module):
    """
    2D CNN over (node × frequency) transfer-function data.

    Input: (batch, 2 * num_nodes, num_freq_bins) — alternating magnitude / phase per node.
    Internally reshaped to (batch, 2, num_nodes, num_freq_bins).
    """

    def __init__(
        self,
        num_nodes: int,
        num_freq_bins: int,
        num_outputs: int,
        pool_h: int = 5,
        pool_w: int = 10,
        conv_channels=(32, 64, 128),
        hidden1: int = 1024,
        hidden2: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_freq_bins = num_freq_bins
        self.num_outputs = num_outputs
        c3 = conv_channels[2]
        flat_dim = c3 * pool_h * pool_w
        self.feature_extractor = _cnn2d_backbone(
            2, pool_h, pool_w, conv_channels=conv_channels
        )
        self.regressor = _regressor_head(
            flat_dim, num_outputs, hidden1, hidden2, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        if x.dim() != 3 or x.size(1) != 2 * self.num_nodes:
            raise ValueError(
                f"Expected (B, {2 * self.num_nodes}, F), got {tuple(x.shape)}"
            )
        x = x.view(batch_size, self.num_nodes, 2, -1).permute(0, 2, 1, 3)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)


class TransferFunction2DCNNMagnitudeOnly(nn.Module):
    """
    Same architecture as TransferFunction2DCNN but only magnitude per node (no phase).

    Input: (batch, num_nodes, num_freq_bins).
    """

    def __init__(
        self,
        num_nodes: int,
        num_freq_bins: int,
        num_outputs: int,
        pool_h: int = 5,
        pool_w: int = 10,
        conv_channels=(32, 64, 128),
        hidden1: int = 1024,
        hidden2: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_freq_bins = num_freq_bins
        self.num_outputs = num_outputs
        c3 = conv_channels[2]
        flat_dim = c3 * pool_h * pool_w
        self.feature_extractor = _cnn2d_backbone(
            1, pool_h, pool_w, conv_channels=conv_channels
        )
        self.regressor = _regressor_head(
            flat_dim, num_outputs, hidden1, hidden2, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != self.num_nodes:
            raise ValueError(
                f"Expected (B, {self.num_nodes}, F), got {tuple(x.shape)}"
            )
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)


class TransferFunction1DCNN(nn.Module):
    """1D CNN on a single node's magnitude and phase along frequency."""

    def __init__(
        self,
        in_channels: int,
        num_freq_bins: int,
        num_outputs: int,
        hidden1: int = 1024,
        hidden2: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_freq_bins = num_freq_bins
        self.num_outputs = num_outputs
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
            nn.Linear(256 * 8, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)


def build_model_from_config(
    kind: str,
    num_nodes: int,
    num_freq_bins: int,
    num_outputs: int,
    arch: dict,
):
    """
    kind: "2d" | "2d_magnitude" | "1d"
    arch: neural_network.architecture + training dropout etc. merged
    """
    pool_h = arch.get("adaptive_pool_height", 5)
    pool_w = arch.get("adaptive_pool_width", 10)
    cc = arch.get("conv_channels", [32, 64, 128])
    conv_ch = tuple(cc) if isinstance(cc, (list, tuple)) and len(cc) == 3 else (32, 64, 128)
    hidden1 = arch.get("regressor_hidden1", 1024)
    hidden2 = arch.get("regressor_hidden2", 512)
    dropout = arch.get("dropout", 0.3)

    if kind == "2d":
        return TransferFunction2DCNN(
            num_nodes=num_nodes,
            num_freq_bins=num_freq_bins,
            num_outputs=num_outputs,
            pool_h=pool_h,
            pool_w=pool_w,
            conv_channels=conv_ch,
            hidden1=hidden1,
            hidden2=hidden2,
            dropout=dropout,
        )
    if kind == "2d_magnitude":
        return TransferFunction2DCNNMagnitudeOnly(
            num_nodes=num_nodes,
            num_freq_bins=num_freq_bins,
            num_outputs=num_outputs,
            pool_h=pool_h,
            pool_w=pool_w,
            conv_channels=conv_ch,
            hidden1=hidden1,
            hidden2=hidden2,
            dropout=dropout,
        )
    if kind == "1d":
        return TransferFunction1DCNN(
            in_channels=2,
            num_freq_bins=num_freq_bins,
            num_outputs=num_outputs,
            hidden1=hidden1,
            hidden2=hidden2,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model kind: {kind}")
