import math
import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.eegnet import Conv2dWithConstraint


class ActSquare(nn.Module):
    """Square activation function."""

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    """
    Logarithm activation function with clamp to avoid log(0).

    Parameters
    ----------
    eps : float, default=1e-6
        Small constant to avoid taking log of zero.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class ADFCNN(EEGModuleMixin, nn.Module):
    """
    Adaptive Deep Frequency Convolutional Neural Network (ADFCNN).

    This model combines spectral and spatial convolutions with attention
    mechanisms for EEG signal classification.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    n_outputs : int
        Number of output classes.
    sampling_rate : int
        Sampling rate of the EEG signals.
    F1 : int, default=8
        Number of spectral filters for the first spectral convolution.
    D : int, default=1
        Depth multiplier for the number of filters.
    F2 : int or str, default='auto'
        Number of spatial filters. If 'auto', F2 = F1 * D.
    pool_mode : str, default='mean'
        Pooling mode, either 'mean' or 'max'.
    dropout : float, default=0.25
        Dropout rate.
    activation : nn.Module, default=nn.ELU
        Activation function class to apply.
    """

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        F1: int = 8,
        D: int = 1,
        F2="auto",
        pool_mode: str = "mean",
        dropout: float = 0.25,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()

        if F2 == "auto":
            F2 = F1 * D

        pooling_layer = {"max": nn.MaxPool2d, "mean": nn.AvgPool2d}[pool_mode]

        # Spectral Convolutions
        self.spectral1 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 125),
                padding="same",
                max_norm=2.0,
            ),
            nn.BatchNorm2d(F1),
        )
        self.spectral2 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 30),
                padding="same",
                max_norm=2.0,
            ),
            nn.BatchNorm2d(F1),
        )

        # Spatial Convolutions
        self.spatial1 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=F2,
                out_channels=F2,
                kernel_size=(n_chans, 1),
                padding=0,
                groups=F2,
                bias=False,
                max_norm=2.0,
            ),
            nn.BatchNorm2d(F2),
            activation(),
            nn.Dropout(dropout),
            Conv2dWithConstraint(
                in_channels=F2,
                out_channels=F2,
                kernel_size=(1, 1),
                padding="valid",
                max_norm=2.0,
            ),
            nn.BatchNorm2d(F2),
            activation(),
            pooling_layer(kernel_size=(1, 32), stride=32),
            nn.Dropout(dropout),
        )

        self.spatial2 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=F2,
                out_channels=F2,
                kernel_size=(n_chans, 1),
                padding="valid",
                max_norm=2.0,
            ),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer(kernel_size=(1, 75), stride=25),
            ActLog(),
            nn.Dropout(dropout),
        )

        # Attention Mechanism
        self.w_q = nn.Linear(F2, F2)
        self.w_k = nn.Linear(F2, F2)
        self.w_v = nn.Linear(F2, F2)
        self.dropout = nn.Dropout(dropout)

        # Final Classification Layer
        self.final_layer = nn.Conv2d(8, n_outputs, kernel_size=(1, 51))

    def forward(self, x):
        """
        Forward pass of the ADFCNN model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, n_times, n_channels).

        Returns
        -------
        torch.Tensor
            Output predictions of shape (batch_size, n_outputs).
        """
        x = x.unsqueeze(1)
        x1 = self.spectral1(x)
        x2 = self.spectral2(x)

        x_filter1 = self.spatial1(x1)
        x_filter2 = self.spatial2(x2)

        x_no_attention = torch.cat((x_filter1, x_filter2), dim=3)
        B2, C2, H2, W2 = x_no_attention.shape
        x_attention = x_no_attention.reshape(B2, C2, H2 * W2).permute(0, 2, 1)

        B, N, C = x_attention.shape

        q = self.w_q(x_attention).permute(0, 2, 1)
        k = self.w_k(x_attention).permute(0, 2, 1)
        v = self.w_v(x_attention).permute(0, 2, 1)

        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)
        d_k = q.size(-1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).reshape(B, N, C)
        x_attention = x_attention + self.dropout(x)
        x_attention = x_attention.reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)

        x = self.dropout(x_attention)
        x = self.final_layer(x)
        x = x.squeeze(3).squeeze(2)

        return x


if __name__ == "__main__":
    x = torch.zeros(1, 22, 1000)
    model = ADFCNN(n_chans=22, n_outputs=2)

    with torch.no_grad():
        out = model(x)
