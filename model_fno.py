import torch
import torch.nn as nn


class DummyModel(nn.Module):
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

            nn.Conv2d(hidden * 2, 48, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out = self.net(x)
        gas = out[:, :24, :, :]
        pressure = out[:, 24:, :, :]
        return gas, pressure