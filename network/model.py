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
        s1, s2, s3, s4, zy, flow_map = self.unet(out)

        return s1, s2, s3, s4, zy, flow_map


class DriverEncoder(nn.Module):
    def __init__(self):
        super(DriverEncoder, self).__init__()

        self.res1 = blocks.ResidualDownBlock(64, 128)
        self.res2 = blocks.ResidualDownBlock(128, 256)
        self.res3 = blocks.ResidualDownBlock(256, 512)
        self.res4 = blocks.ResidualDownBlock(512, 512)

    def forward(self, drv_lmark):
        out = self.res1(drv_lmark)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        return out


class Blender(nn.Module):
    def __init__(self):
        super(Blender, self).__init__()

    def forward(self, zx, ):
        # 3 image attention blocks
        pass


class Decoder(nn.Module):
    def __init__(self, im_size):
        super(Decoder, self).__init__()

        self.warp1 = blocks.WarpAlignBlock(im_size, im_size)
        self.warp2 = blocks.WarpAlignBlock(im_size, im_size)
        self.warp3 = blocks.WarpAlignBlock(im_size, im_size)
        self.warp4 = blocks.WarpAlignBlock(im_size, im_size)

    def forward(self, z_xy, s1, s2, s3, s4):
        # 4 warp-alignment blocks
        # TODO: maybe in up-down direction
        u = self.warp1(s1, z_xy)
        u = self.warp1(s2, u)
        u = self.warp1(s3, u)
        u = self.warp1(s4, u)
        return u


class Generator(nn.Module):
    def __init__(self, in_height):
        super(Generator, self).__init__()

        self.target_encoder = TargetEncoder(in_height)
        self.driver_encoder = DriverEncoder()
        self.blender = Blender()
        self.decoder = Decoder(in_height)

    def forward(self, drv_lmark, target_imgs, target_lmarks):
        s1, s2, s3, s4, zy = self.target_encoder(target_imgs, target_lmarks)
        zx = self.driver_encoder(drv_lmark)
        z_xy = self.blender(zx, zy)
        result = self.decoder(z_xy, s1, s2, s3, s4)

        return result


class Discriminator(nn.Module):
    def __init__(self, num_videos, batch_size):
        super(Discriminator, self).__init__()
        pass

    def forward(self, x, y, i):
        pass
