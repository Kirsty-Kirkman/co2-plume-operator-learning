import torch
import torch.nn as nn
import torch.fft


class SpectralConv2d(nn.Module):
    """
    2D spectral convolution layer for FNO.

    Applies FFT -> learned complex weights on low-frequency modes -> inverse FFT.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # input:   (batch, in_channels, x, y)
        # weights: (in_channels, out_channels, x, y)
        # output:  (batch, out_channels, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x = x.shape[-2]
        size_y = x.shape[-1]

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            size_x,
            size_y // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, size_x)
        m2 = min(self.modes2, size_y // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2],
        )

        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, :m1, :m2],
        )

        x = torch.fft.irfft2(out_ft, s=(size_x, size_y))
        return x


class FNOBlock2d(nn.Module):
    """
    One FNO block:
    spectral conv + pointwise conv + activation.
    """

    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.spectral(x) + self.w(x)
        x = self.act(x)
        return x


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for CCS plume prediction.

    Input shape:
        (batch, in_ch, nz, nx)

    Output:
        gas:      (batch, 24, nz, nx)
        pressure: (batch, 24, nz, nx)

    Notes
    -----
    - Set in_ch=11 if using:
      [porosity, perm_r, perm_z, inj_rate, temperature, depth,
       Swi, lam, perf_mask, z_channel, r_channel]
    - Predicts all 24 timesteps at once for both fields.
    """

    def __init__(
        self,
        in_ch=11,
        width=48,
        modes1=12,
        modes2=16,
        n_blocks=4,
        out_ch_per_field=24,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.width = width
        self.out_ch_per_field = out_ch_per_field

        # Lift input channels to model width
        self.input_proj = nn.Conv2d(in_ch, width, kernel_size=1)

        # FNO blocks
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes1, modes2) for _ in range(n_blocks)]
        )

        # Shared decoder trunk
        self.decoder = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
        )

        # Separate heads
        self.gas_head = nn.Conv2d(width, out_ch_per_field, kernel_size=1)
        self.pressure_head = nn.Conv2d(width, out_ch_per_field, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.decoder(x)

        gas = self.gas_head(x)
        pressure = self.pressure_head(x)

        return gas, pressure