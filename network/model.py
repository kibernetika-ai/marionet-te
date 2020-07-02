import math

import torch
import torch.nn as nn

from network import blocks
from network import unet

E_LEN = 512


class TargetEncoder(nn.Module):
    def __init__(self, image_size=256):
        super(TargetEncoder, self).__init__()

        self.unet = unet.UNet(n_channels=6, n_classes=2)

    def forward(self, img, lmark):
        out = torch.cat((img, lmark), dim=-3)  # out 6*224*224
        s1, s2, s3, s4, zy = self.unet(out)

        return s1, s2, s3, s4, zy


class DriverEncoder(nn.Module):
    def __init__(self):
        super(DriverEncoder, self).__init__()

        self.res1 = blocks.ResidualDownBlock(3, 64, norm=nn.InstanceNorm2d)
        self.res2 = blocks.ResidualDownBlock(64, 128, norm=nn.InstanceNorm2d)
        self.res3 = blocks.ResidualDownBlock(128, 256, norm=nn.InstanceNorm2d)
        self.res4 = blocks.ResidualDownBlock(256, 512, norm=nn.InstanceNorm2d)

    def forward(self, drv_lmark):
        out = self.res1(drv_lmark)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        return out


class Blender(nn.Module):
    def __init__(self):
        super(Blender, self).__init__()

        self.att1 = blocks.ImageAttention(d_model=512, n_channels=64)
        self.att2 = blocks.ImageAttention(d_model=512, n_channels=64)
        self.att3 = blocks.ImageAttention(d_model=512, n_channels=64)

    def forward(self, zx, zy):
        # zx: [1, 512, 16, 16]
        # zy: [B*8, 512, 8, 8]
        # 3 image attention blocks
        z_xy = self.att1(zx, zy)
        z_xy = self.att1(z_xy, zy)
        out = self.att1(z_xy, zy)

        return out


class Decoder(nn.Module):
    def __init__(self, channels=512):
        super(Decoder, self).__init__()

        self.warp1 = blocks.WarpAlignBlock(channels, channels // 2)
        self.warp2 = blocks.WarpAlignBlock(channels // 2, channels // 4)
        self.warp3 = blocks.WarpAlignBlock(channels // 4, channels // 8)
        self.warp4 = blocks.WarpAlignBlock(channels // 8, channels // 16)

        # last conv with 3 channels to get the image
        self.conv = nn.Conv2d(channels, 3, 1, 1)

    def forward(self, z_xy, s1, s2, s3, s4):
        # z_xy: [B, 512, 16, 16]
        # s1: [BxK, 128, 128, 128]
        # s2: [BxK, 256, 64, 64]
        # s3: [BxK, 512, 32, 32]
        # s4: [BxK, 512, 16, 16]
        # TODO: maybe in up-down direction
        u = self.warp1(s4, z_xy)
        u = self.warp1(s3, u)
        u = self.warp1(s2, u)
        u = self.warp1(s1, u)

        out = self.conv(u)
        out = torch.tanh(out)
        return out


class Generator(nn.Module):
    def __init__(self, in_height, emb_dim=512):
        super(Generator, self).__init__()

        self.target_encoder = TargetEncoder(in_height)
        self.driver_encoder = DriverEncoder()
        self.blender = Blender()
        self.decoder = Decoder(emb_dim)

    def forward(self, drv_lmark, target_imgs, target_lmarks):
        s1, s2, s3, s4, zy = self.target_encoder(target_imgs, target_lmarks)
        zx = self.driver_encoder(drv_lmark)
        z_xy = self.blender(zx, zy)
        result = self.decoder(z_xy, s1, s2, s3, s4)

        return result


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.res1 = blocks.ResidualDownBlock(64, 128, norm=None)
        self.res2 = blocks.ResidualDownBlock(128, 256, norm=None)
        self.res3 = blocks.ResidualDownBlock(256, 512, norm=None)
        self.res4 = blocks.ResidualDownBlock(512, 512, norm=None)
        self.res5 = blocks.ResidualDownBlock(512, 512, norm=None)

    def forward(self, x, r, c):
        pass
