import torch
from torch import nn


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DownConv, self).__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UpConv, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        x_dec = self.upsample(x_dec)

        x = torch.cat([x_enc, x_dec], dim=1)  # cat across channels

        return self.conv(x)


class UNet(nn.Module):
    def __init__(self) -> None:
        super(UNet, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.enc1 = DownConv(64, 128)
        self.enc2 = DownConv(128, 256)
        self.enc3 = DownConv(256, 256)
        # self.enc4 = DownConv(512, 512)
        # self.dec1 = UpConv(1024, 256)
        self.dec2 = UpConv(512, 128)
        self.dec3 = UpConv(256, 64)
        self.dec4 = UpConv(128, 64)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        out = self.enc3(x3)
        # x4 = self.enc3(x3)
        # x5 = self.enc4(x4)
        # out = self.dec1(x4, x5)
        out = self.dec2(x3, out)
        out = self.dec3(x2, out)
        out = self.dec4(x1, out)
        out = self.out_conv(out)
        out = self.sigmoid(out)

        return out


# https://box.skoltech.ru/index.php/s/ib52BOoV58ztuPM
class HGDownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(HGDownConv, self).__init__()
        self.downsample = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


class HGUpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(HGUpConv, self).__init__()
        self.upsample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class SkipConnection(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(SkipConnection, self).__init__()
        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip_connection(x)


class HourGlass(nn.Module):
    depth = 5

    def __init__(self) -> None:
        super(HourGlass, self).__init__()
        encs = []
        decs = []
        skips = []

        enc_out_channels = 128
        dec_in_channels = 132
        dec_out_channels = 128
        for i in range(HourGlass.depth):
            enc_in_channels = 32 if i == 0 else 128
            encs.append(HGDownConv(enc_in_channels, enc_out_channels))
            decs.append(HGUpConv(dec_in_channels, dec_out_channels))
            skips.append(SkipConnection(128, 4))

        self.encs = nn.ModuleList(encs)
        self.decs = nn.ModuleList(decs)
        self.skips = nn.ModuleList(skips)
        self.out_conv = nn.Conv2d(dec_out_channels, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_outs = []
        enc_out = x
        for i in range(HourGlass.depth):
            enc_out = self.encs[i](enc_out)
            skip_outs.append(self.skips[i](enc_out))

        dec_out = torch.cat((enc_out, skip_outs[-1]), dim=1)
        for i in range(HourGlass.depth, 0, -1):
            dec_out = self.decs[i - 1](dec_out)
            if i > 1:
                skip_out = skip_outs[i - 2]
                dec_out = torch.cat((dec_out, skip_out), dim=1)

        dec_out = self.out_conv(dec_out)
        dec_out = self.sigmoid(dec_out)
        return dec_out
