# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: Apache-2.0
# Adapted from https://github.com/wzhlearning/fNIRSNet (Apache-2.0).
"""fNIRSNet: Rethinking Delayed Hemodynamic Responses for fNIRS Classification."""

from __future__ import annotations

from collections import OrderedDict

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import _disable_batch_norm_training_if_batch_size_one
from braindecode.modules import DepthwiseConv2d


class FNIRSNet(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""fNIRSNet from Wang et al. (2023) :footcite:`wang2023fnirsnet`.

    :bdg-success:`Convolution`

    .. versionadded:: 2.1

    From the paper: "fNIRSNet consists of a delayed hemodynamic response module
    (DHR Module) and a global module. Specifically, fNIRSNet contains three
    convolutional layers with different kernel sizes and a fully connected (FC)
    layer. Delayed hemodynamic response convolutions (DHRConv) extract the
    channel-level features of delayed hemodynamic responses. Depthwise separable
    convolutions consisting of depthwise convolutions (DWConv) and pointwise
    convolutions (PWConv) are used to reduce model parameters. Batch
    normalization (BN) accelerates network training and improves classification
    performance. The sigmoid activation function is used for nonlinear
    activation and alleviates information loss in the first layer. The feature
    maps are flattened to 1D vectors and then fed into an FC layer."

    The input is the paper's fNIRS tensor, :math:`C` rows (``n_chans``: the HbO
    and HbR signals of every fNIRS channel) by :math:`S` sampling points
    (``n_times``). The DHR module is ``conv1`` (DHRConv, kernel
    :math:`1 \times S`) and ``bn1``; the global module is ``conv2.depth_conv``
    (DWConv, kernel :math:`C \times 1`), ``conv2.point_conv`` (PWConv,
    :math:`1 \times 1`) and ``bn2``; ``final_layer`` is the FC layer. The
    parameter names are the authors', and ``mapping`` loads their ``fc.*``
    weights, so their ``state_dict`` loads with ``strict=True``.

    `License <https://github.com/wzhlearning/fNIRSNet/blob/main/LICENSE>`_

    Parameters
    ----------
    n_dhr_filters : int, default=4
        Number of DHRConv filters (:math:`F_1` in the paper, ``num_DHRConv`` in
        the authors' code).
    n_dw_filters : int, default=8
        Number of filters of the global module (:math:`F_2` in the paper,
        ``num_DWConv`` in the authors' code).
    dhr_kernel_length : int or None, default=None
        Width of the DHRConv kernel (``DHRConv_width``). ``None`` uses
        :math:`S` = ``n_times``: "the width of this kernel equals the number of
        input sampling points" (the paper's Guideline 1).
    activation : type[nn.Module], default=nn.Sigmoid
        Activation after each batch normalization (sigmoid in the paper).

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        n_dhr_filters: int = 4,
        n_dw_filters: int = 8,
        dhr_kernel_length: int | None = None,
        activation: type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__(
            n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        kernel_length = dhr_kernel_length or self.n_times
        self.to_fnirs_tensor = Rearrange("batch chans time -> batch 1 chans time")
        self.conv1 = nn.Conv2d(1, n_dhr_filters, (1, kernel_length))
        self.bn1 = nn.BatchNorm2d(n_dhr_filters)
        self.conv2 = nn.Sequential(
            OrderedDict(
                depth_conv=DepthwiseConv2d(n_dhr_filters, 1, (self.n_chans, 1)),
                point_conv=nn.Conv2d(n_dhr_filters, n_dw_filters, 1),
            )
        )
        self.bn2 = nn.BatchNorm2d(n_dw_filters)
        self.activation = activation()
        self.flatten = Rearrange("batch filters 1 time -> batch (filters time)")
        n_features = n_dw_filters * (self.n_times - kernel_length + 1)
        self.final_layer = nn.Linear(n_features, self.n_outputs)
        self.mapping = {f"fc.{k}": f"final_layer.{k}" for k in ("weight", "bias")}

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    @_disable_batch_norm_training_if_batch_size_one
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Logits ``(batch, n_outputs)`` of ``x`` ``(batch, n_chans, n_times)``.

        ``return_features=True`` gives ``{"features", "cls_token": None}``; a
        ``ch_mask`` raises a ``ValueError`` (DWConv spans all channels).
        """
        if ch_mask is not None:
            raise ValueError("FNIRSNet does not accept ch_mask: DWConv spans n_chans.")
        # time' = n_times - dhr_kernel_length + 1 (1 with the default kernel)
        x = self.to_fnirs_tensor(x)  # (batch, 1, n_chans, n_times)
        x = self.conv1(x)  # DHRConv: (batch, n_dhr_filters, n_chans, time')
        x = self.bn1(x)  # (batch, n_dhr_filters, n_chans, time')
        x = self.activation(x)  # (batch, n_dhr_filters, n_chans, time')
        x = self.conv2(x)  # DWConv, PWConv: (batch, n_dw_filters, 1, time')
        x = self.bn2(x)  # (batch, n_dw_filters, 1, time')
        x = self.activation(x)  # (batch, n_dw_filters, 1, time')
        features = self.flatten(x)  # (batch, n_dw_filters * time')
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)  # (batch, n_outputs)
