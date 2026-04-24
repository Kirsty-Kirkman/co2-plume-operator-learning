import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    CNN baseline for CO₂ plume prediction with a shared encoder
    and separate output heads for:
    - gas saturation
    - pressure buildup

    Input shape:
        (batch, in_ch, nz, nx)

    Output shapes:
        gas:      (batch, 24, nz, nx)
        pressure: (batch, 24, nz, nx)
    """

    def __init__(self, in_ch=11, hidden=64):
        super().__init__()

        # Shared feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Gas saturation head
        self.gas_head = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 24, kernel_size=3, padding=1),
        )

        # Pressure buildup head
        self.pressure_head = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 24, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_ch, nz, nx)

        Returns
        -------
        gas : torch.Tensor
            Predicted gas saturation with shape (batch, 24, nz, nx)

        pressure : torch.Tensor
            Predicted pressure buildup with shape (batch, 24, nz, nx)
        """
        if x.ndim != 4:
            raise ValueError(
                "BaselineCNN expects input shape (batch, channels, nz, nx). "
                f"Received tensor with shape {tuple(x.shape)}."
            )

        expected_channels = self.encoder[0].in_channels
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"BaselineCNN expected {expected_channels} input channels, got {x.shape[1]}."
            )

        features = self.encoder(x)

        gas = self.gas_head(features)
        pressure = self.pressure_head(features)

        return gas, pressure