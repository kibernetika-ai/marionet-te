import torch
import torch.nn as nn


class WarpAlignBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(WarpAlignBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, s, u):
        f_u = self.conv1(u)
        pose_adapt = nn.functional.grid_sample(s, f_u.permute([0, 2, 3, 1]))

        out = torch.cat([f_u, pose_adapt])
        out = self.conv2(out)
        out = self.upsample(out)
        return out


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualDownBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, stride=2):
        downsample = nn.Sequential(
            conv3x3(self.in_channels, out_channels, stride=2),
            nn.BatchNorm2d(out_channels)
        )
        super(ResidualDownBlock, self).__init__(in_channels, out_channels, stride=stride, downsample=downsample)


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
    def __init__(self, in_channels, out_channels, is_bilinear=True):
        super(ResidualBlockUp, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

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


class ResBlockUp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size=None, scale=2, conv_size=3, padding_size=1, is_bilinear=True):
        super(ResBlockUp, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        if is_bilinear:
            self.upsample = nn.Upsample(size=out_size, scale_factor=scale, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample(size=out_size, scale_factor=scale)
        self.relu = nn.LeakyReLU(inplace=False)

        # left
        self.conv_l1 = nn.Conv2d(in_channel, out_channel, 1)

        # right
        self.conv_r1 = nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size)
        self.conv_r2 = nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size)

    def forward(self, x):
        res = x

        # left
        out_res = self.upsample(res)
        out_res = self.conv_l1(out_res)

        # right
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.relu(out)
        out = self.conv_r2(out)

        out = out + out_res

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
