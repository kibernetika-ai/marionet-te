import math

import torch
import torch.nn as nn


class WarpAlignBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(WarpAlignBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=1, bias=False)
        self.res_up = ResidualBlockUp(out_channel, out_channel // 2, norm=nn.InstanceNorm2d)

    def forward(self, s, u):
        f_u = self.conv1(u)
        pose_adapt = nn.functional.grid_sample(s, f_u.permute([0, 2, 3, 1]))

        out = torch.cat([f_u, pose_adapt])
        out = self.conv2(out)
        out = self.res_up(out)

        return out


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def positionalencoding2d(channels, height, width):
    """
    :param channels: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if channels % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(channels))
    pe = torch.zeros(channels, height, width)
    # Each dimension use half of d_model
    channels = int(channels / 2)
    div_term = torch.exp(torch.arange(0., channels, 2) *
                         -(math.log(10000.0) / channels))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:channels:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:channels:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[channels::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[channels + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class SinusoidalEncoding(nn.Module):
    def __init__(self):
        super(SinusoidalEncoding, self).__init__()

    def forward(self, x):
        return x


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        if norm:
            if norm is nn.InstanceNorm2d:
                self.bn1 = norm(out_channels, affine=True)
                self.bn2 = norm(out_channels, affine=True)
            else:
                self.bn1 = norm(out_channels)
                self.bn2 = norm(out_channels)
        else:
            self.bn1 = None
            self.bn2 = None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.bn1:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.bn2:
            out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualDownBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, stride=2, norm=nn.BatchNorm2d):
        downsample = nn.Sequential(
            conv3x3(self.in_channels, out_channels, stride=2),
        )
        if norm is not None:
            if norm is nn.InstanceNorm2d:
                downsample.add_module('1', norm(out_channels, affine=True))
            else:
                downsample.add_module('1', norm(out_channels))
        super(ResidualDownBlock, self).__init__(
            in_channels, out_channels, stride=stride, downsample=downsample, norm=norm
        )


class ImageAttention(nn.Module):
    def __init__(self, im_size):
        super(ImageAttention, self).__init__()

        self.conv3x3 = conv3x3(im_size, im_size)
        self.inst_norm1 = nn.InstanceNorm2d(im_size, affine=True)
        self.inst_norm2 = nn.InstanceNorm2d(im_size, affine=True)

    def forward(self, q, k, v):
        # res = attention
        # out = res + q
        # res = instance norm
        # out = conv3x3
        # out = out + res
        # out = instance norm
        pass


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()

        # conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        # conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        # conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))

        self.softmax = nn.Softmax(-2)  # sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x)  # BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x)  # BxC'xHxW
        h_projection = self.conv_h(x)  # BxCxHxW

        f_projection = torch.transpose(f_projection.view(B, -1, H * W), 1, 2)  # BxNxC', N=H*W
        g_projection = g_projection.view(B, -1, H * W)  # BxC'xN
        h_projection = h_projection.view(B, -1, H * W)  # BxCxN

        attention_map = torch.bmm(f_projection, g_projection)  # BxNxN
        attention_map = self.softmax(attention_map)  # sum_i_N (A i,j) = 1

        # sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map)  # BxCxN
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


# Residual block
class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, is_bilinear=True, norm=nn.BatchNorm2d):
        super(ResidualBlockUp, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        if norm is not None:
            if norm is nn.InstanceNorm2d:
                self.bn1 = norm(out_channels, affine=True)
                self.bn2 = norm(out_channels, affine=True)
                self.bn3 = norm(out_channels, affine=True)
            else:
                self.bn1 = norm(out_channels)
                self.bn2 = norm(out_channels)
                self.bn3 = norm(out_channels)
        else:
            self.bn1 = self.bn2 = self.bn3 = None

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)

        if is_bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        residual = x
        # left
        out_res = self.upsample(x)
        out_res = self.conv1(out_res)
        out_res = self.bn1(out_res)

        # right
        out = self.upsample(residual)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out = out + out_res
        out = self.relu(out)

        return out


class Padding(nn.Module):
    def __init__(self, in_shape):
        super(Padding, self).__init__()

        self.zero_pad = nn.ZeroPad2d(self.findPadSize(in_shape))

    def forward(self, x):
        out = self.zero_pad(x)
        return out

    def findPadSize(self, in_shape):
        if in_shape < 256:
            pad_size = (256 - in_shape) // 2
        else:
            pad_size = 0
        return pad_size
