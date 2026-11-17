# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
# Adapted from https://github.com/wzhlearning/fNIRS-Transformer (MIT).
"""fNIRS-T and fNIRS-PreT: Transformer Models for fNIRS Classification."""

from __future__ import annotations

from typing import Literal

import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import fnirs_channel_order
from braindecode.modules import (
    FeedForwardBlock,
    MultiHeadAttention,
    PreNormTransformerBlock,
)


class FNIRST(EEGModuleMixin, nn.Module, license="mit"):
    r"""fNIRS-T from Wang et al. (2022).

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    .. versionadded:: 2.1

    .. figure:: https://braindecode.org/dev/_static/model/fnirst.png
       :align: center
       :alt: fNIRS-T and fNIRS-PreT architecture

       Figure of the authors' repository (``fig/model.png``,
       wzhlearning/fNIRS-Transformer at ``6ebe987f``, MIT): the
       :math:`H \times W \times C` input (channels, samples, HbO/HbR) feeds a
       "Spatial-level" and a "Channel-level" branch. Bottom: fNIRS-T takes
       filtered, baseline-corrected signals; fNIRS-PreT adds a
       "Preprocessing Module" (:class:`FNIRSPreT`). Braindecode returns logits.

    Wang et al. propose "an fNIRS classification network based on
    Transformer, named fNIRS-T", and "explore the spatial-level and
    channel-level representation of fNIRS signals to improve data utilization
    and network representation capacity" :footcite:`wang2022fnirst`. In the
    figure's symbols, :math:`\mathrm{Conv}_S` (``to_patch_embedding``) gives
    :math:`H_S` = ``H - spatial_kernel_size + 1`` tokens, one per group of
    neighbouring channels, and :math:`\mathrm{Conv}_C`
    (``to_channel_embedding``) :math:`H_C = H` tokens, one per channel; each
    token of :math:`W_S \times C_S` features is projected to ``dim`` and
    normalized. Each branch adds a class token and a learned positional
    embedding and runs ``depth`` pre-norm transformer blocks with bias-free
    q/k/v; the two pooled vectors are concatenated, normalized
    (``final_norm``) and classified (``final_layer``, the authors'
    ``mlp_head``).

    ============================  =========================  ====================
    Class                         Input in the authors'      First module
                                  pipeline
    ============================  =========================  ====================
    ``FNIRST`` (fNIRS-T)          band-pass filtered,        none
                                  baseline-corrected
    :class:`FNIRSPreT`            raw (no filtering or       ``pre``: average
                                  baseline correction)       pooling, LayerNorm
    ============================  =========================  ====================

    **Braindecode mapping.** ``x`` is ``(batch, n_chans, n_times)``,
    ``n_chans = 2 * H``: HbO rows then HbR rows, or MNE's interleaved order
    read from ``chs_info``; ``to_chromophore_planes`` makes the authors'
    ``[B, 2, channels, samples]``. The convolutions count samples, so
    ``sfreq`` is unused; no trained weights are released. The positional
    tables have one row per token (the authors allocate 101 rows and slice),
    so an authors' ``state_dict`` loads after a conversion, not a ``mapping``:
    ``to_qkv`` split into ``queries``/``keys``/``values``, tables cut,
    ``mlp_head.{0,1}`` to ``final_norm``/``final_layer``. The port then
    matches the authors' model to float32 rounding (max-abs <= 3.3e-07).
    The attention always projects its output (the authors skip it when
    ``heads=1`` and ``dim_head=dim``). ``n_chromophores=1``, the clamp of
    ``spatial_kernel_size`` and ``mask`` are additions of the port;
    ``ch_mask`` is refused, as every layer spans the whole montage.

    `License <https://github.com/wzhlearning/fNIRS-Transformer/blob/main/LICENSE>`_

    Parameters
    ----------
    dim : int, default=64
        Authors' ``dim``, "last dimension of output tensor after linear
        transformation"; 64 for their datasets A and B, 128 for C.
    depth : int, default=6
        Authors' ``depth``, "number of Transformer blocks" per branch.
    heads : int, default=8
        Authors' ``heads``, "number of the multi-head self-attention".
    mlp_dim : int, default=64
        Authors' ``mlp_dim``, "dimension of the MLP layer".
    pool : {"cls", "mean"}, default="cls"
        Authors' ``pool``: "'cls' is [CLS] token pooling, 'mean' is average
        pooling", per branch.
    dim_head : int, default=64
        Authors' ``dim_head``: width of each attention head.
    conv_channels : int, default=8
        :math:`C_S = C_C`, feature maps of both convolutions (authors: 8).
    spatial_kernel_size : int, default=5
        Height of :math:`\mathrm{Conv}_S` in channels (authors: 5), clamped to
        the number of channels.
    temporal_kernel_size : int, default=30
        Temporal kernel of both convolutions, in samples (authors: 30);
        ``n_times`` must be at least this.
    temporal_stride : int, default=4
        Temporal stride of both convolutions, in samples (authors: 4).
    n_chromophores : {1, 2}, default=2
        :math:`C`. ``2`` (HbO/HbR, the authors' input) needs an even
        ``n_chans``; ``1`` is an addition of the port.
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_H, HbR_1..HbR_H]`` or MNE's ``[HbO_1, HbR_1, ...]``;
        ``None`` reads it from ``chs_info``.
    drop_prob : float, default=0.0
        Authors' ``dropout`` in the attention and feed-forward sub-layers.
    emb_drop_prob : float, default=0.0
        Authors' ``emb_dropout``, "dropout for patch embeddings".
    activation : type[nn.Module], default=nn.GELU
        Activation of the feed-forward sub-layers (authors: GELU).

    Notes
    -----
    Protocol of the authors' released code (the paper is not open access).
    Datasets: A (mental arithmetic vs rest, 8 subjects, 52 channels, 140
    samples), B (Shin et al. 2017, same task, 29 subjects, 36 channels, 200
    samples), C (Bak et al. 2019, finger and foot tapping, 3 classes, 30
    subjects, 20 channels, 256 samples). fNIRS-T inputs are band-passed (B, C:
    3rd-order Butterworth 0.01-0.1 Hz, ``filtfilt``), baseline-corrected and
    z-scored per window. Training: 5 x 5-fold cross-validation over pooled
    trials or leave-one-subject-out, 120 epochs, batch 128, AdamW, cosine
    annealing (``T_max=30``), label smoothing 0.1 (k-fold), flooding (0.45,
    0.40, 0.35 from epochs 1, 30, 50); the best test-fold accuracy is kept.

    Replication on dataset C with the authors' ``KFold_Train.py`` (seeds 0-2,
    mean of the 25 runs; macro-F1 at the kept epoch):

    ==========  =================  ================  ================
    Model       Paper (abstract)   Port (acc / F1)   Authors' code
    ==========  =================  ================  ================
    fNIRS-T     75.49 %            75.03 % / 0.7498  74.73 % / 0.7469
    fNIRS-PreT  "competitive"      72.50 % / 0.7244  72.50 % / 0.7245
    ==========  =================  ================  ================

    References
    ----------
    .. footbibliography::
    """

    _preblock = False  # FNIRSPreT prepends the preprocessing module

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        dim: int = 64,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 64,
        pool: Literal["cls", "mean"] = "cls",
        dim_head: int = 64,
        conv_channels: int = 8,
        spatial_kernel_size: int = 5,
        temporal_kernel_size: int = 30,
        temporal_stride: int = 4,
        n_chromophores: int = 2,
        channel_order: Literal["stacked", "interleaved"] | None = None,
        drop_prob: float = 0.0,
        emb_drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if self.n_times < temporal_kernel_size:
            raise ValueError(
                f"n_times ({self.n_times}) must be >= temporal_kernel_size."
            )
        if n_chromophores not in (1, 2) or self.n_chans % n_chromophores:
            raise ValueError(
                f"n_chromophores must be 1 or 2 and divide n_chans ({self.n_chans}), "
                f"got {n_chromophores}."
            )
        pairs_info = self._chs_info if n_chromophores == 2 else None
        order = fnirs_channel_order(pairs_info, channel_order)
        channel_axes = "(chans chrom)" if order == "interleaved" else "(chrom chans)"
        self.to_chromophore_planes = Rearrange(
            f"batch {channel_axes} time -> batch chrom chans time",
            chrom=n_chromophores,
        )
        n_phys_chans = self.n_chans // n_chromophores
        self.pool = pool
        n_bins = (self.n_times - temporal_kernel_size) // temporal_stride + 1
        spatial_kernel = min(spatial_kernel_size, n_phys_chans)
        self.pre = _PreBlock(self.n_times, n_chromophores) if self._preblock else None

        def branch() -> nn.Sequential:
            return nn.Sequential(
                *[
                    PreNormTransformerBlock(
                        dim,
                        MultiHeadAttention(
                            dim, heads, head_dim=dim_head, qkv_bias=False
                        ),
                        FeedForwardBlock(
                            dim, 1, drop_prob, activation, hidden_features=mlp_dim
                        ),
                        drop_prob,
                    )
                    for _ in range(depth)
                ]
            )

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(
                n_chromophores,
                conv_channels,
                (spatial_kernel, temporal_kernel_size),
                (1, temporal_stride),
            ),
            Rearrange("batch maps tokens bins -> batch tokens (maps bins)"),
            nn.Linear(n_bins * conv_channels, dim),
            nn.LayerNorm(dim),
        )
        self.cls_token_patch = nn.Parameter(torch.empty(1, 1, dim))
        self.pos_embedding_patch = nn.Parameter(
            torch.empty(1, n_phys_chans - spatial_kernel + 2, dim)
        )
        self.dropout_patch = nn.Dropout(emb_drop_prob)
        self.transformer_patch = branch()

        self.to_channel_embedding = nn.Sequential(
            nn.Conv2d(
                n_chromophores,
                conv_channels,
                (1, temporal_kernel_size),
                (1, temporal_stride),
            ),
            Rearrange("batch maps tokens bins -> batch tokens (maps bins)"),
            nn.Linear(n_bins * conv_channels, dim),
            nn.LayerNorm(dim),
        )
        self.cls_token_channel = nn.Parameter(torch.empty(1, 1, dim))
        self.pos_embedding_channel = nn.Parameter(torch.empty(1, n_phys_chans + 1, dim))
        self.dropout_channel = nn.Dropout(emb_drop_prob)
        self.transformer_channel = branch()

        self.token_mean = Reduce("batch tokens dim -> batch dim", "mean")
        self.final_norm = nn.LayerNorm(dim * 2)
        self.final_layer = nn.Linear(dim * 2, self.n_outputs)
        self._init_parameters()

    def _init_parameters(self) -> None:
        # Drawn after the layers, so reset_parameters() repeats the constructor.
        with torch.no_grad():
            self.cls_token_patch.normal_()
            self.pos_embedding_patch.normal_()
            self.cls_token_channel.normal_()
            self.pos_embedding_channel.normal_()

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """``x``: ``(batch, n_chans, n_times)``. ``mask`` (bool, same shape):
        True samples are zeroed first. ``ch_mask`` is refused: the spatial
        kernel and positional tables span every channel. ``return_features``
        gives the normalized pooled ``"features"`` and, as ``"cls_token"``, the
        two branches' class tokens ``(batch, 2 * dim)``."""
        if ch_mask is not None:
            raise ValueError("FNIRST does not accept ch_mask (fixed channel layout).")
        if mask is not None:
            x = x.masked_fill(mask, 0.0)  # (batch, n_chans, n_times)
        # (batch, n_chans, n_times) -> (batch, n_chromophores, n_phys_chans, n_times)
        x = self.to_chromophore_planes(x)
        if self.pre is not None:
            x = self.pre(x)  # (batch, n_chromophores, n_phys_chans, n_times)
        batch_size = x.shape[0]
        # spatial level, n_groups = n_phys_chans - spatial_kernel_size + 1 tokens
        patch = self.to_patch_embedding(x)  # (batch, n_groups, dim)
        cls_patch = self.cls_token_patch.expand(batch_size, -1, -1)  # (batch, 1, dim)
        patch = torch.cat((cls_patch, patch), dim=1)  # (batch, n_groups + 1, dim)
        patch = patch + self.pos_embedding_patch  # (batch, n_groups + 1, dim)
        patch = self.dropout_patch(patch)  # (batch, n_groups + 1, dim)
        patch = self.transformer_patch(patch)  # (batch, n_groups + 1, dim)
        # channel level, n_tokens = n_phys_chans + 1
        channel = self.to_channel_embedding(x)  # (batch, n_phys_chans, dim)
        cls_channel = self.cls_token_channel.expand(batch_size, -1, -1)
        channel = torch.cat((cls_channel, channel), dim=1)  # (batch, n_tokens, dim)
        channel = channel + self.pos_embedding_channel  # (batch, n_tokens, dim)
        channel = self.dropout_channel(channel)  # (batch, n_tokens, dim)
        channel = self.transformer_channel(channel)  # (batch, n_tokens, dim)
        patch_token = patch[:, 0]  # (batch, dim)
        channel_token = channel[:, 0]  # (batch, dim)
        cls_tokens = torch.cat((patch_token, channel_token), dim=1)  # (batch, 2 * dim)
        if self.pool == "mean":
            patch = self.token_mean(patch)  # (batch, dim)
            channel = self.token_mean(channel)  # (batch, dim)
            features = torch.cat((patch, channel), dim=1)  # (batch, 2 * dim)
        else:
            features = cls_tokens
        features = self.final_norm(features)  # (batch, 2 * dim)
        if return_features:
            return {"features": features, "cls_token": cls_tokens}
        return self.final_layer(features)  # (batch, n_outputs)


class FNIRSPreT(FNIRST, license="mit"):  # the kwarg is not inherited
    r"""fNIRS-PreT from Wang et al. (2022).

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    .. versionadded:: 2.1

    .. figure:: https://braindecode.org/dev/_static/model/fnirst.png
       :align: center
       :alt: fNIRS-T and fNIRS-PreT architecture

       The figure of the authors' repository (``fig/model.png`` of
       wzhlearning/fNIRS-Transformer at ``6ebe987f``, MIT). fNIRS-PreT is the
       right-hand input path: "Segmentation", then the "Preprocessing Module",
       then the two branches of fNIRS-T. Braindecode returns the logits.

    From the abstract :footcite:`wang2022fnirst`: "a preprocessing module,
    which consists of one-dimensional average pooling and layer normalization,
    is designed to replace filtering and baseline correction of data
    preprocessing. It makes fNIRS-T an end-to-end network, called fNIRS-PreT."
    The abstract adds that "fNIRS-PreT using raw data also achieves
    competitive performance to fNIRS-T".

    This is :class:`~braindecode.models.FNIRST` with that preprocessing module
    (``pre``) first, as the authors' ``fNIRS_PreT`` (``PreBlock``, then
    ``fNIRS_T``):

    - ``pre.smooth``: three moving averages along time (``AvgPool1d`` with
      kernels 5, 13 and 7, stride 1, "same" padding), shared by HbO and HbR
      (the authors' ``pool1``, ``pool2``, ``pool3``);
    - ``pre.norms.0`` and ``pre.norms.1``: one LayerNorm over the
      ``n_times`` samples per chromophore (the authors' ``ln_0``, ``ln_1``),
      so its weights are tied to ``n_times``.

    The parameters, the input layout (stacked or interleaved HbO/HbR), the
    opt-in ``mask`` and the head are those of
    :class:`~braindecode.models.FNIRST`. An authors' ``fNIRS_PreT``
    ``state_dict`` loads after FNIRST's conversion, with the ``fNIRS_T.``
    prefix removed and ``pre.ln_i`` renamed ``pre.norms.i``; the port and the
    authors' model then agree to float32 rounding (max-abs at most 3.3e-07
    for the dataset A, B and C configurations).

    `License <https://github.com/wzhlearning/fNIRS-Transformer/blob/main/LICENSE>`_

    Notes
    -----
    The protocol of the authors' released code is that of
    :class:`~braindecode.models.FNIRST`, on raw data: their README asks to
    disable the filtering and baseline correction of ``B_mat2xls.m`` and
    ``C_mat2xls.m`` by hand for fNIRS-PreT, and ``Load_Dataset_A`` skips the
    dataset A filters for ``model='fNIRS-PreT'``. The flooding levels are
    those of fNIRS-T except on dataset B (0.40, 0.38 and 0.35 from epochs 1,
    30 and 50).

    The abstract gives no fNIRS-PreT accuracy. In the authors'
    ``KFold_Train.py`` on dataset C (5 x 5-fold, seeds 0, 1 and 2, the metric
    of :class:`~braindecode.models.FNIRST`), the port reaches 72.50 %
    accuracy (macro-F1 0.7244) and the authors' model 72.50 % (0.7245). Both
    received the band-pass filtered, baseline-corrected windows of fNIRS-T:
    the released ``Load_Dataset_C`` has no raw-data path.

    References
    ----------
    .. footbibliography::
    """

    _preblock = True


class _PreBlock(nn.Module):
    """The authors' preprocessing module: moving averages, then LayerNorm over time."""

    def __init__(self, n_times: int, n_chromophores: int) -> None:
        super().__init__()
        self.smooth = nn.Sequential(  # "same" padding
            nn.AvgPool1d(5, stride=1, padding=2),
            nn.AvgPool1d(13, stride=1, padding=6),
            nn.AvgPool1d(7, stride=1, padding=3),
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(n_times) for _ in range(n_chromophores)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        planes = []
        for chromophore, norm in enumerate(self.norms):
            plane = x[:, chromophore]  # (batch, n_phys_chans, n_times)
            plane = self.smooth(plane)  # (batch, n_phys_chans, n_times)
            plane = norm(plane)  # (batch, n_phys_chans, n_times)
            planes.append(plane)
        # (batch, n_chromophores, n_phys_chans, n_times)
        return torch.stack(planes, dim=1)
