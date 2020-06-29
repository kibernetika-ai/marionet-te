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

    def forward(self, img):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, img):
        pass


class Generator(nn.Module):
    def __init__(self, in_height):
        super(Generator, self).__init__()

        self.target_encoder = TargetEncoder(in_height)
        self.driver_encoder = DriverEncoder()
        self.blender = Blender()
        self.decoder = Decoder()

    def forward(self, img, target_imgs, target_lmarks):
        s1, s2, s3, s4, zy = self.target_encoder(target_imgs, target_lmarks)


class Discriminator(nn.Module):
    def __init__(self, num_videos, batch_size):
        super(Discriminator, self).__init__()
        pass

    def forward(self, x, y, i):
        pass
