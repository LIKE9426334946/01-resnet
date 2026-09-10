import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=16,
    ):
        super().__init__()

        # =========================
        # Encoder
        # =========================

        self.encoder1 = ConvBlock(
            in_channels,
            base_channels,
        )

        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = ConvBlock(
            base_channels,
            base_channels * 2,
        )

        self.pool2 = nn.MaxPool2d(2)

        # =========================
        # Bottleneck
        # =========================

        self.bottleneck = ConvBlock(
            base_channels * 2,
            base_channels * 4,
        )

        # =========================
        # Decoder
        # =========================

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )

        self.decoder2 = ConvBlock(
            base_channels * 4,
            base_channels * 2,
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )

        self.decoder1 = ConvBlock(
            base_channels * 2,
            base_channels,
        )

        # =========================
        # Output
        # =========================

        self.output = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        # =========================
        # Encoder
        # =========================

        e1 = self.encoder1(x)

        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)

        p2 = self.pool2(e2)

        # =========================
        # Bottleneck
        # =========================

        b = self.bottleneck(p2)

        # =========================
        # Decoder 2
        # =========================

        d2 = self.up2(b)

        # Skip Connection
        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )

        d2 = self.decoder2(d2)

        # =========================
        # Decoder 1
        # =========================

        d1 = self.up1(d2)

        # Skip Connection
        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )

        d1 = self.decoder1(d1)

        # =========================
        # Output
        # =========================

        output = self.output(d1)

        return output
