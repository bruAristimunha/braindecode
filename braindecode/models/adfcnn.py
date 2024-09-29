import math
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin
from braindecode.models.eegnet import Conv2dWithConstraint
from braindecode.models.modules import SafeLog, Square


class _SelfAttentionLayer(nn.Module):
    """
    Self-Attention Layer for the ADFCNN Model.

    [As defined above]
    """

    def __init__(self, attention_dim: int, dropout: float = 0.25):
        super(_SelfAttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        self.dropout_layer = nn.Dropout(dropout)

        self.w_q = nn.Linear(attention_dim, attention_dim)
        self.w_k = nn.Linear(attention_dim, attention_dim)
        self.w_v = nn.Linear(attention_dim, attention_dim)

    def forward(self, x_filter1: torch.Tensor, x_filter2: torch.Tensor) -> torch.Tensor:
        # [As defined above]
        x_no_attention = torch.cat(
            (x_filter1, x_filter2), dim=3
        )  # Shape: (B2, C2, H2, W2)

        B2, C2, H2, W2 = x_no_attention.shape
        # Reshape and permute for attention computation
        x_attention = x_no_attention.reshape(B2, C2, H2 * W2).permute(
            0, 2, 1
        )  # Shape: (B2, H2*W2, C2)

        B, N, C = x_attention.shape  # B: batch size, N: sequence length, C: channels

        # Compute queries, keys, and values
        q = self.w_q(x_attention).permute(0, 2, 1)  # Shape: (B, C, N)
        k = self.w_k(x_attention).permute(0, 2, 1)  # Shape: (B, C, N)
        v = self.w_v(x_attention).permute(0, 2, 1)  # Shape: (B, C, N)

        # Normalize queries and keys
        q = nn.functional.normalize(q, dim=-1)  # Shape: (B, C, N)
        k = nn.functional.normalize(k, dim=-1)  # Shape: (B, C, N)

        d_k = q.size(-1)
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)  # Shape: (B, C, C)
        attn = attn.softmax(dim=-1)  # Shape: (B, C, C)

        # Apply attention to values
        x = (attn @ v).reshape(B, N, C)  # Shape: (B, N, C)

        # Residual connection with dropout
        x_attention = x_attention + self.dropout_layer(x)  # Shape: (B, N, C)

        # Reshape back to original tensor shape
        x_attention = x_attention.reshape(B2, H2, W2, C2).permute(
            0, 3, 1, 2
        )  # Shape: (B2, C2, H2, W2)

        return x_attention


class ADFCNN(EEGModuleMixin, nn.Module):
    """
    Adaptive Deep Frequency Convolutional Neural Network (ADFCNN).

    XXXXX.

    Parameters
    ----------
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
        n_chans=None,
        n_outputs=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        F1: int = 8,
        D: int = 1,
        F2="auto",
        pool_mode: str = "mean",
        dropout: float = 0.25,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        if F2 == "auto":
            F2 = F1 * D

        pooling_layer = {"max": nn.MaxPool2d, "mean": nn.AvgPool2d}[pool_mode]

        self.ensuredims = Rearrange(
            " batch ntimes nchannels -> batch 1 nchannels ntimes"
        )

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
                kernel_size=(self.n_chans, 1),
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
                kernel_size=(self.n_chans, 1),
                padding="valid",
                max_norm=2.0,
            ),
            nn.BatchNorm2d(F2),
            Square(),
            pooling_layer(kernel_size=(1, 75), stride=25),
            SafeLog(),
            nn.Dropout(dropout),
        )
        self.attention_layer = _SelfAttentionLayer(attention_dim=F2, dropout=dropout)
        # Attention Mechanism
        self.w_q = nn.Linear(F2, F2)
        self.w_k = nn.Linear(F2, F2)
        self.w_v = nn.Linear(F2, F2)
        self.dropout = nn.Dropout(dropout)

        # Final Classification Layer
        self.final_layer = nn.Conv2d(F1, self.n_outputs, kernel_size=(1, 51))

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

        x_attention = self.attention_layer(
            x_filter1, x_filter2
        )  # Shape: (B, F2, H2, W2)

        out = self.final_layer(x_attention)
        out = torch.squeeze(out, 3)
        out = torch.squeeze(out, 2)
        return out


if __name__ == "__main__":
    x = torch.zeros(1, 22, 1000)
    model = ADFCNN(n_chans=22, n_outputs=2)

    with torch.no_grad():
        out = model(x)
    print(out.shape)
