import torch
import torch.nn as nn

class FCBlock(nn.Module):

    def __init__(self, in_dim, out_dim, batch_norm, dropout) -> None:
        super().__init__()

        self.layer = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim) if batch_norm else lambda x: x
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.layer(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding,
    pool_type, batch_norm, dropout) -> None:

        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.pool = (
            nn.AvgPool2d(kernel_size, stride) if pool_type == "AvgPool"
            else nn.MaxPool2d(kernel_size, stride)
        )
        self.batch_norm = nn.BatchNorm2d(out_ch) if batch_norm else lambda x: x
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):

        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class Conv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding,
    pool_type, batch_norm, dropout) -> None:

        super().__init__()
        self.conv_layer = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.pool = (
            nn.AvgPool3d(kernel_size, stride) if pool_type == "AvgPool"
            else nn.MaxPool3d(kernel_size, stride)
        )
        self.batch_norm = nn.BatchNorm3d(out_ch) if batch_norm else lambda x: x
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):

        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x
