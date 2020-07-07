import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2
        self.down4 = Down(512, 1024 // factor)
        self.down5 = Down(512, 1024 // factor)

        self.avg_pool1 = nn.AvgPool2d(2)
        self.avg_pool2 = nn.AvgPool2d(4)
        self.avg_pool3 = nn.AvgPool2d(8)

        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128 // factor, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        s1 = self.down1(x1)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        s5 = self.down5(s4)
        x = self.up1(s5, s4)  # 512 / 512
        x = self.up2(x, s3)  # 256 / 256
        x = self.up3(x, s2)  # 128 / 128
        x = self.up4(x, s1)  # 64 / 64

        flow_map = self.outc(x)
        flow_map = torch.tanh(flow_map)

        fy2 = self.avg_pool1(flow_map)
        fy3 = self.avg_pool2(flow_map)
        fy4 = self.avg_pool3(flow_map)

        # Apply warping function
        s1 = nn.functional.grid_sample(s1, flow_map.permute([0, 2, 3, 1]))
        s2 = nn.functional.grid_sample(s2, fy2.permute([0, 2, 3, 1]))
        s3 = nn.functional.grid_sample(s3, fy3.permute([0, 2, 3, 1]))
        s4 = nn.functional.grid_sample(s4, fy4.permute([0, 2, 3, 1]))

        return s1, s2, s3, s4, s5


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.utils.spectral_norm(
                nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=1, stride=2)
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        return self.conv(x)
