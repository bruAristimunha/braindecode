import random
from typing import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn


class Net(nn.Module):
    # what does F_out mean here? final output feature dimension?
    # how can i control the final output classes?
    # what is outchans here?
    # what is D2 here?

    def __init__(self, F_out, outchans, K, D2=320, number_conv=5):
        super().__init__()
        ch_locs = [ch["loc"] for ch in self.chs_info]  # type: ignore

        self.D2 = D2
        self.outchans = outchans
        self.spatial_attention = SpatialAttention(out_channels=outchans, K=K)
        self.conv = nn.Conv2d(outchans, outchans, 1, padding="same")
        self.conv_blocks = nn.Sequential(
            *[self.generate_conv_block(k) for k in range(number_conv)]
        )  # 5 conv blocks
        self.final_convs = nn.Sequential(
            nn.Conv2d(self.D2, self.D2 * 2, 1),
            nn.GELU(),
            nn.Conv2d(self.D2 * 2, F_out, 1),
        )
        self.l1 = nn.Linear(256 * F_out, 2)
        self.rearrange = Rearrange("batch chan time -> batch 1 chan time")

    def generate_conv_block(self, k):
        kernel_size = (1, 3)
        padding = "same"
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            self.outchans if k == 0 else self.D2,
                            self.D2,
                            kernel_size,
                            dilation=pow(2, (2 * k) % 5),
                            padding=padding,
                        ),
                    ),
                    ("bn1", nn.BatchNorm2d(self.D2)),
                    ("gelu1", nn.GELU()),
                    (
                        "conv2",
                        nn.Conv2d(
                            self.D2,
                            self.D2,
                            kernel_size,
                            dilation=pow(2, (2 * k + 1) % 5),
                            padding=padding,
                        ),
                    ),
                    ("bn2", nn.BatchNorm2d(self.D2)),
                    ("gelu2", nn.GELU()),
                    (
                        "conv3",
                        nn.Conv2d(self.D2, self.D2 * 2, kernel_size, padding=padding),
                    ),
                    ("glu", nn.GLU(dim=1)),
                ]
            )
        )

    def forward(self, x):
        x = self.rearrange(x)
        # x shape: N x 1 x chan x time
        x = self.spatial_attention(x[:, 0]).unsqueeze(
            2
        )  # add dummy dimension at the end
        x = self.conv(x)

        for k in range(len(self.conv_blocks)):
            if k == 0:
                x = self.conv_blocks[k](x)
            else:
                x_copy = x
                for name, module in self.conv_blocks[k].named_modules():
                    if name == "conv2" or name == "conv3":
                        x = (
                            x_copy + x
                        )  # residual skip connection for the first two convs
                        x_copy = x.clone()  # is it deep copy?
                    x = module(x)
        x = self.final_convs(x)
        x = torch.flatten(x, 1)
        x = F.softmax(self.l1(x), -1)

        return x


class SpatialAttention(nn.Module):
    def __init__(self, out_channels, K):
        super().__init__()
        self.outchans = out_channels
        self.K = K
        # trainable parameter:
        self.z = nn.Parameter(
            torch.randn(self.outchans, K * K, dtype=torch.cfloat) / (32 * 32),
            requires_grad=True,
        )  # each output channel has its own KxK z matrix

    def forward(self, X):
        # How can i generate this dynamically based on input chan number, and only once?
        # what is necessary to make this work with batch processing?
        cos_mat24, sin_mat24 = compute_cos_sin(
            positions24[:, 0], positions24[:, 1], self.K
        )
        cos_mat128, sin_mat128 = compute_cos_sin(
            positions128[:, 0], positions128[:, 1], self.K
        )
        a_24 = torch.matmul(self.z.real, cos_mat24.T) + torch.matmul(
            self.z.imag, sin_mat24.T
        )
        a_128 = torch.matmul(self.z.real, cos_mat128.T) + torch.matmul(
            self.z.imag, sin_mat128.T
        )
        sol = []
        index = random.randint(0, 22)
        x_drop = positions24[index, 0]
        y_drop = positions24[index, 1]
        # this for is really necessary?
        #
        # How can we batch this operation?
        for eeg in X:
            if eeg.shape[0] == 23:
                a = a_24
                x_coord = positions24[:, 0]
                y_coord = positions24[:, 1]
            elif eeg.shape[0] == 128:
                a = a_128
                x_coord = positions128[:, 0]
                y_coord = positions128[:, 1]
            # Question: divide this with square root of KxK? to stabilize gradient as with self-attention?
            for i in range(a.shape[1]):
                distance = (x_drop - x_coord[i]) ** 2 + (y_drop - y_coord[i]) ** 2
                if distance < 0.1:
                    # Drop channels that are close to the randomly selected one
                    a = torch.cat((a[:, :i], a[:, i + 1 :]), dim=1)
                    eeg = torch.cat((eeg[:i], eeg[i + 1 :]), dim=0)

            a = F.softmax(
                a, dim=1
            )  # softmax over all input chan location for each output chan
            # outchans x  inchans

            # X: N x 273 x 360
            sol.append(torch.matmul(a, eeg))  # N x outchans x 360 (time)
            # matmul dim expansion logic: https://pytorch.org/docs/stable/generated/torch.matmul.html
        return torch.stack(sol)


def compute_cos_sin(x, y, K):
    kk = torch.arange(1, K + 1)
    ll = torch.arange(1, K + 1)
    cos_fun = lambda k, l, x, y: torch.cos(2 * torch.pi * (k * x + l * y))
    sin_fun = lambda k, l, x, y: torch.sin(2 * torch.pi * (k * x + l * y))
    return torch.stack(
        [cos_fun(kk[None, :], ll[:, None], x, y) for x, y in zip(x, y)]
    ).reshape(x.shape[0], -1).float(), torch.stack(
        [sin_fun(kk[None, :], ll[:, None], x, y) for x, y in zip(x, y)]
    ).reshape(x.shape[0], -1).float()


positions128 = torch.tensor(
    np.array(
        [
            [7.15185774e-01, 2.48889351e-01],
            [7.02434125e-01, 3.17666701e-01],
            [6.95043285e-01, 3.81730395e-01],
            [6.64632066e-01, 4.15512025e-01],
            [6.17343244e-01, 4.52640363e-01],
            [5.70554706e-01, 4.86034445e-01],
            [5.21555270e-01, 5.11613217e-01],
            [7.77079001e-01, 3.33967050e-01],
            [7.58449443e-01, 4.04282683e-01],
            [7.24088505e-01, 4.36604693e-01],
            [6.85433073e-01, 4.86034445e-01],
            [6.17343244e-01, 5.19428527e-01],
            [5.59172750e-01, 5.38580522e-01],
            [8.18418971e-01, 4.41881185e-01],
            [7.60133440e-01, 4.86034445e-01],
            [7.35627966e-01, 4.86034445e-01],
            [8.45364169e-01, 4.86034445e-01],
            [7.24088505e-01, 5.35464197e-01],
            [6.64632066e-01, 5.56556865e-01],
            [6.08030772e-01, 5.74970128e-01],
            [8.18418971e-01, 5.30187705e-01],
            [7.58449443e-01, 5.67786207e-01],
            [6.95043285e-01, 5.90338495e-01],
            [6.39581248e-01, 5.97577739e-01],
            [7.77079001e-01, 6.38101840e-01],
            [7.02434125e-01, 6.54402189e-01],
            [6.36080215e-01, 6.39322188e-01],
            [5.90689607e-01, 6.21729782e-01],
            [5.49930209e-01, 5.93637909e-01],
            [5.09601084e-01, 5.65410534e-01],
            [4.74660346e-01, 5.26921639e-01],
            [7.15185774e-01, 7.23179539e-01],
            [6.27432502e-01, 7.00712475e-01],
            [5.70168887e-01, 6.72695237e-01],
            [5.24983262e-01, 6.41061825e-01],
            [4.95401265e-01, 6.09669558e-01],
            [4.56179857e-01, 5.70442673e-01],
            [6.52314440e-01, 7.82311284e-01],
            [5.18992954e-01, 7.39998487e-01],
            [4.89867494e-01, 6.91482721e-01],
            [4.69348952e-01, 6.55201509e-01],
            [4.34670981e-01, 6.24309562e-01],
            [6.28180126e-01, 8.61450930e-01],
            [5.49739502e-01, 8.11172169e-01],
            [4.24802980e-01, 7.37204171e-01],
            [4.25680921e-01, 6.87383067e-01],
            [4.13285406e-01, 6.55344129e-01],
            [6.71360349e-01, 9.72068890e-01],
            [5.16948449e-01, 9.08970332e-01],
            [3.55532101e-01, 7.10977843e-01],
            [3.62120543e-01, 6.64285008e-01],
            [3.79156415e-01, 6.28501244e-01],
            [3.98210730e-01, 5.80314761e-01],
            [4.15632463e-01, 5.34804692e-01],
            [4.44251810e-01, 4.86034445e-01],
            [3.37112089e-01, 8.67523508e-01],
            [3.26773432e-01, 7.71448657e-01],
            [3.02409518e-01, 6.81637463e-01],
            [3.15334354e-01, 6.26408474e-01],
            [3.42865950e-01, 5.85266696e-01],
            [3.65550361e-01, 5.38836376e-01],
            [3.34656872e-01, 4.86034445e-01],
            [2.28916019e-01, 8.08418208e-01],
            [2.40816811e-01, 7.14482456e-01],
            [2.54299126e-01, 6.35765016e-01],
            [2.82959322e-01, 5.83008874e-01],
            [3.09968299e-01, 5.30931543e-01],
            [1.33841406e-01, 6.96028404e-01],
            [1.74061857e-01, 6.32194751e-01],
            [2.13486999e-01, 5.73685443e-01],
            [2.63433137e-01, 5.23590932e-01],
            [2.93992338e-01, 4.86034445e-01],
            [8.69873448e-02, 5.89501183e-01],
            [1.46141772e-01, 5.31677906e-01],
            [2.05894177e-01, 4.86034445e-01],
            [2.63433137e-01, 4.48477958e-01],
            [3.09968299e-01, 4.41137347e-01],
            [3.65550361e-01, 4.33232515e-01],
            [4.15632463e-01, 4.37264198e-01],
            [4.74660346e-01, 4.45147251e-01],
            [7.87721494e-02, 4.86034445e-01],
            [1.46141772e-01, 4.40390984e-01],
            [2.13486999e-01, 3.98383447e-01],
            [2.82959322e-01, 3.89060016e-01],
            [3.42865950e-01, 3.86802194e-01],
            [3.98210730e-01, 3.91754129e-01],
            [4.56179857e-01, 4.01626217e-01],
            [8.69873448e-02, 3.82567707e-01],
            [1.74061857e-01, 3.39874139e-01],
            [2.54299126e-01, 3.36303874e-01],
            [3.15334354e-01, 3.45660416e-01],
            [3.79156415e-01, 3.43567646e-01],
            [4.34670981e-01, 3.47759328e-01],
            [1.33841406e-01, 2.76040486e-01],
            [2.40816811e-01, 2.57586434e-01],
            [3.02409518e-01, 2.90431427e-01],
            [3.62120543e-01, 3.07783882e-01],
            [4.13285406e-01, 3.16724761e-01],
            [2.28916019e-01, 1.63650682e-01],
            [3.26773432e-01, 2.00620233e-01],
            [3.55532101e-01, 2.61091048e-01],
            [4.25680921e-01, 2.84685823e-01],
            [4.69348952e-01, 3.16867381e-01],
            [4.95401265e-01, 3.62399332e-01],
            [5.09601084e-01, 4.06658357e-01],
            [5.21555270e-01, 4.60455673e-01],
            [3.37112089e-01, 1.04545383e-01],
            [4.24802980e-01, 2.34864719e-01],
            [4.89867494e-01, 2.80586169e-01],
            [5.24983262e-01, 3.31007065e-01],
            [5.49930209e-01, 3.78430981e-01],
            [5.59172750e-01, 4.33488368e-01],
            [5.16948449e-01, 6.30985576e-02],
            [5.49739502e-01, 1.60896721e-01],
            [5.18992954e-01, 2.32070403e-01],
            [5.70168887e-01, 2.99373653e-01],
            [5.90689607e-01, 3.50339108e-01],
            [6.08030772e-01, 3.97098762e-01],
            [6.71360349e-01, -5.20417043e-18],
            [6.28180126e-01, 1.10617960e-01],
            [6.52314440e-01, 1.89757606e-01],
            [6.27432502e-01, 2.71356415e-01],
            [6.36080215e-01, 3.32746702e-01],
            [6.39581248e-01, 3.74491151e-01],
            [7.15997599e-01, 1.78986800e-01],
            [9.02096460e-01, 2.53445652e-01],
            [9.02096460e-01, 7.18623238e-01],
            [7.15997599e-01, 7.93082090e-01],
        ]
    )
)

positions24 = positions128[
    [
        22,
        9,
        33,
        24,
        11,
        124,
        122,
        29,
        6,
        111,
        45,
        36,
        104,
        108,
        42,
        55,
        93,
        58,
        52,
        62,
        92,
        96,
        70,
    ]
]


if __name__ == "__main__":
    Model = Net(F_out=120, outchans=270, K=3)

    data = torch.zeros((2, 23, 256))  # this is the correct shape?
    with torch.no_grad():
        out = Model(data)
        print(out)
