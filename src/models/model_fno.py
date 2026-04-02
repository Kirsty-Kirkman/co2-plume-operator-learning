import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Simple CNN baseline for CO₂ plume prediction.

    This model maps input reservoir/property channels to
    time-dependent output fields for:
    - gas saturation
    - pressure buildup

    Input shape:
        (batch, in_ch, nz, nx)

    Output shapes:
        gas:      (batch, 24, nz, nx)
        pressure: (batch, 24, nz, nx)
    """

    def __init__(self, in_ch=9, hidden=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),

            # 48 output channels = 24 gas + 24 pressure
            nn.Conv2d(hidden * 2, 48, kernel_size=3, padding=1),
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
        out = self.net(x)

        # out shape: (batch, 48, nz, nx)
        gas = out[:, :24, :, :]
        pressure = out[:, 24:, :, :]

        return gas, pressure