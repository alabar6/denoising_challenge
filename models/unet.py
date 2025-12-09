import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- building blocks ----------
class DoubleConv(nn.Module):
    """(Conv → ReLU) × 2 with padding to keep H×W unchanged."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Max-pool then double conv."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c)
        )

    def forward(self, x):
        return self.block(x)


# class Up(nn.Module):
#     """Up-sample (transposed conv) → concat with encoder feat → double conv."""
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.up   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
#         self.conv = DoubleConv(in_c, out_c)  # in_c = out_c (upsampled) + out_c (encoder)

#     def forward(self, x, enc_feat):
#         x = self.up(x)
#         x = torch.cat([enc_feat, x], dim=1)
#         return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = DoubleConv(in_c, out_c)  # после concat: 2*out_c -> out_c

    def forward(self, x, enc_feat):
        # Вариант A: строго подогнать через interpolate
        x = self.up(x)
        if (x.size(2), x.size(3)) != (enc_feat.size(2), enc_feat.size(3)):
            x = F.interpolate(x, size=enc_feat.shape[-2:],
                              mode='bilinear', align_corners=False)
        x = torch.cat([enc_feat, x], dim=1)
        return self.conv(x)


# ---------- full U-Net ----------
class UNetSameSize(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1):
        super().__init__()
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64,   128)
        self.down2 = Down(128,  256)
        self.down3 = Down(256,  512)
        self.down4 = Down(512,  1024)

        self.up1   = Up(1024, 512)
        self.up2   = Up(512,  256)
        self.up3   = Up(256,  128)
        self.up4   = Up(128,  64)

        self.outc  = nn.Conv2d(64, n_classes, kernel_size=1)

        # Kaiming initialization (He et al.)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)  # logits; apply sigmoid/softmax externally

