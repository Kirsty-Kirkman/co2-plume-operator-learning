import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.ParameterList(
            [
                nn.Parameter(
                    scale * torch.randn(
                        in_channels,
                        out_channels,
                        modes1,
                        modes2,
                        modes3,
                        dtype=torch.cfloat,
                    )
                )
                for _ in range(4)
            ]
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        T = x.size(-3)
        Z = x.size(-2)
        Rf = x.size(-1) // 2 + 1

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            T,
            Z,
            Rf,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, T)
        m2 = min(self.modes2, Z)
        m3 = min(self.modes3, Rf)

        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.weights[0][:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, :m2, :m3], self.weights[1][:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, -m2:, :m3], self.weights[2][:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, -m2:, :m3], self.weights[3][:, :, :m1, :m2, :m3]
        )

        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


class FNO3d(nn.Module):
    def __init__(self, in_ch=11, width=32, modes_t=8, modes_z=12, modes_r=12):
        super().__init__()
        self.input_proj = nn.Conv3d(in_ch, width, kernel_size=1)

        self.spectral = SpectralConv3d(width, width, modes_t, modes_z, modes_r)
        self.bypass = nn.Conv3d(width, width, kernel_size=1)

        self.gas_head = nn.Sequential(
            nn.Conv3d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.pres_head = nn.Sequential(
            nn.Conv3d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = F.gelu(self.spectral(x) + self.bypass(x))
        gas = self.gas_head(x)
        pres = self.pres_head(x)
        return gas, pres