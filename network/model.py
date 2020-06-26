import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlockDown, SelfAttention, ResBlock, ResBlockD, ResBlockUp, Padding, adaIN
import math
import sys
import os

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

    def forward(self, img):
        pass


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

    def forward(self, img, target_imgs, target_lmarks):
        s1, s2, s3, s4, zy = self.target_encoder(target_imgs, target_lmarks)


class Discriminator(nn.Module):
    def __init__(self, num_videos, batch_size, finetuning=False, e_finetuning=None):
        super(Discriminator, self).__init__()
        self.relu = nn.LeakyReLU()

        # in 6*224*224
        self.pad = Padding(224)  # out 6*256*256
        self.resDown1 = ResBlockDown(6, 64)  # out 64*128*128
        self.resDown2 = ResBlockDown(64, 128)  # out 128*64*64
        self.resDown3 = ResBlockDown(128, 256)  # out 256*32*32
        self.self_att = SelfAttention(256)  # out 256*32*32
        self.resDown4 = ResBlockDown(256, 512)  # out 512*16*16
        self.resDown5 = ResBlockDown(512, 512)  # out 512*8*8
        self.resDown6 = ResBlockDown(512, E_LEN)  # out 512*4*4
        self.res = ResBlockD(E_LEN)  # out 512*4*4
        self.sum_pooling = nn.AdaptiveAvgPool2d((1, 1))  # out 512*1*1

        self.W_i = nn.Parameter(torch.randn(E_LEN, num_videos))
        self.w_0 = nn.Parameter(torch.randn(E_LEN, 1))
        self.b = nn.Parameter(torch.randn(1))

        self.finetuning = finetuning
        self.e_finetuning = e_finetuning
        self.w_prime = nn.Parameter(torch.randn(E_LEN, 1))

    def finetuning_init(self):
        if self.finetuning:
            self.w_prime = nn.Parameter(self.w_0 + self.e_finetuning.mean(dim=0))

    def load_W_i(self, W_i):
        self.W_i.data[:, :W_i.shape[1]] = self.relu(W_i)

    def forward(self, x, y, i):
        out = torch.cat((x, y), dim=-3)  # out B*6*224*224

        out = self.pad(out)

        out1 = self.resDown1(out)

        out2 = self.resDown2(out1)

        out3 = self.resDown3(out2)

        out = self.self_att(out3)

        out4 = self.resDown4(out)

        out5 = self.resDown5(out4)

        out6 = self.resDown6(out5)

        out7 = self.res(out6)

        out = self.sum_pooling(out7)

        out = self.relu(out)

        out = out.squeeze(-1)  # out B*512*1

        if self.finetuning:
            out = torch.bmm(out.transpose(1, 2), (self.w_prime.unsqueeze(0).expand(out.shape[0], 512, 1))) + self.b
        else:
            out = torch.bmm(
                out.transpose(1, 2),
                (self.W_i[:, i].unsqueeze(-1)).transpose(0, 1) + self.w_0
            ) + self.b  # 1x1

        return out, [out1, out2, out3, out4, out5, out6, out7]


class Cropped_VGG19(nn.Module):
    def __init__(self):
        super(Cropped_VGG19, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.conv3_3 = nn.Conv2d(256, 256, 3)
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        self.conv4_3 = nn.Conv2d(512, 512, 3)
        self.conv5_1 = nn.Conv2d(512, 512, 3)
        # self.conv5_2 = nn.Conv2d(512,512,3)
        # self.conv5_3 = nn.Conv2d(512,512,3)

    def forward(self, x):
        conv1_1_pad = F.pad(x, (1, 1, 1, 1))
        conv1_1 = self.conv1_1(conv1_1_pad)
        relu1_1 = F.relu(conv1_1)
        conv1_2_pad = F.pad(relu1_1, (1, 1, 1, 1))
        conv1_2 = self.conv1_2(conv1_2_pad)
        relu1_2 = F.relu(conv1_2)
        pool1_pad = F.pad(relu1_2, (0, 1, 0, 1), value=float('-inf'))
        pool1 = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2_1_pad = F.pad(pool1, (1, 1, 1, 1))
        conv2_1 = self.conv2_1(conv2_1_pad)
        relu2_1 = F.relu(conv2_1)
        conv2_2_pad = F.pad(relu2_1, (1, 1, 1, 1))
        conv2_2 = self.conv2_2(conv2_2_pad)
        relu2_2 = F.relu(conv2_2)
        pool2_pad = F.pad(relu2_2, (0, 1, 0, 1), value=float('-inf'))
        pool2 = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv3_1_pad = F.pad(pool2, (1, 1, 1, 1))
        conv3_1 = self.conv3_1(conv3_1_pad)
        relu3_1 = F.relu(conv3_1)
        conv3_2_pad = F.pad(relu3_1, (1, 1, 1, 1))
        conv3_2 = self.conv3_2(conv3_2_pad)
        relu3_2 = F.relu(conv3_2)
        conv3_3_pad = F.pad(relu3_2, (1, 1, 1, 1))
        conv3_3 = self.conv3_3(conv3_3_pad)
        relu3_3 = F.relu(conv3_3)
        pool3_pad = F.pad(relu3_3, (0, 1, 0, 1), value=float('-inf'))
        pool3 = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv4_1_pad = F.pad(pool3, (1, 1, 1, 1))
        conv4_1 = self.conv4_1(conv4_1_pad)
        relu4_1 = F.relu(conv4_1)
        conv4_2_pad = F.pad(relu4_1, (1, 1, 1, 1))
        conv4_2 = self.conv4_2(conv4_2_pad)
        relu4_2 = F.relu(conv4_2)
        conv4_3_pad = F.pad(relu4_2, (1, 1, 1, 1))
        conv4_3 = self.conv4_3(conv4_3_pad)
        relu4_3 = F.relu(conv4_3)
        pool4_pad = F.pad(relu4_3, (0, 1, 0, 1), value=float('-inf'))
        pool4 = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv5_1_pad = F.pad(pool4, (1, 1, 1, 1))
        conv5_1 = self.conv5_1(conv5_1_pad)
        relu5_1 = F.relu(conv5_1)

        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]
