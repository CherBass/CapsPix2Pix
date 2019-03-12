# modified from https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(args, in_channels, out_channels, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.dconv_down1 = double_conv(args, 1, 32)
        self.dconv_down2 = double_conv(args,32, 64)
        self.dconv_down3 = double_conv(args,64, 128)
        self.dconv_down4 = double_conv(args,128, 256)

        self.dconv_down5 = double_conv(args,256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dropout = nn.Dropout2d(args['drop_out'])
        self.dconv_up4 = double_conv(args,256 + 512, 256)
        self.dconv_up3 = double_conv(args,128 + 256, 128)
        self.dconv_up2 = double_conv(args,128 + 64, 64)
        self.dconv_up1 = double_conv(args,64 + 32, 32)

        self.conv_last = nn.Conv2d(32, args['n_class'], 1)

    def forward(self, x, args):
        conv1 = self.dconv_down1(x)
        if args['drop_out']:
            conv1 = self.dropout(conv1)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        if args['drop_out']:
            conv2 = self.dropout(conv2)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        if args['drop_out']:
            conv3 = self.dropout(conv3)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        if args['drop_out']:
            conv4 = self.dropout(conv4)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        if args['drop_out']:
            x = self.dropout(conv5)


        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        if args['drop_out']:
            x = self.dropout(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        if args['drop_out']:
            x = self.dropout(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        if args['drop_out']:
            x = self.dropout(x)
        x = self.upsample(x)

        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        if args['drop_out']:
            x = self.dropout(x)

        out = F.sigmoid(self.conv_last(x))

        return out
