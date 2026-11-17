# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3-Clause
# Adapted from https://github.com/GeoNIRS/GeoNIRS (81f8995; no licence file;
# paper arXiv:2610.17845).

from __future__ import annotations

import math
from collections import OrderedDict
from typing import List, Literal

import numpy as np
import torch
from torch import nn

from braindecode.functional import (
    check_ch_mask,
    pad_or_crop_to_patches,
    patch_count,
    scaled_dot_product_attention,
)
from braindecode.functional.patches import _check_on_non_divisible
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import fnirs_channel_order, fnirs_split_hbo_hbr


class GeoNIRS(EEGModuleMixin, nn.Module):
    r"""GeoNIRS from Li et al. (2026) :footcite:`li2026geonirs`.

    :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    GeoNIRS is "a geometry-aware pretraining framework that uses the physical
    locations of source-detector pairs as a shared spatial reference across
    heterogeneous probe layouts. GeoNIRS compresses variable-size channel sets
    into a fixed-size latent representation" :footcite:`li2026geonirs`. This
    class is its encoder with a linear head; the cross-chromophore
    reconstruction decoders of pretraining are not part of it.

    **Temporal and geometric representations.** The paired chromophore
    signals are :math:`X^O, X^R \in \mathbb{R}^{C \times T}`. "Along the
    temporal dimension, each channel is divided into :math:`P` non-overlapping
    patches, each containing :math:`S` samples. Every patch is projected into
    a :math:`D`-dimensional token and augmented with a learnable temporal
    positional embedding. Separate HbO and HbR temporal encoders then model
    the token sequence within each physical channel, producing
    :math:`H^O, H^R \in \mathbb{R}^{C \times P \times D}`." Each channel is
    represented by the six-dimensional geometry vector
    :math:`g_c = [\mathrm{mid}_c, \Delta_c]`, with
    :math:`\mathrm{mid}_c = (s_c + d_c) / 2` and :math:`\Delta_c = d_c - s_c`
    for its source :math:`s_c` and detector :math:`d_c`, centred on the montage
    and divided by "a fixed reference scale of 25.46 mm"; a geometry encoder
    (an MLP and one Transformer layer) maps it to :math:`G`.

    **Geometry-aware channel compression.** GeoNIRS introduces ":math:`M`
    learnable latent queries, denoted by :math:`L \in \mathbb{R}^{M \times D}`.
    For the :math:`p`-th temporal patch, the paired HbO and HbR
    representations are first fused as
    :math:`\bar{H}_p = \frac{1}{2}(H^O_{:,p} + H^R_{:,p})`", and

    .. math::

        Q = W_Q \mathrm{LN}(L), \quad
        K_p = W_K \mathrm{LN}(\bar{H}_p + G), \quad
        A_p = \mathrm{softmax}\left(\frac{Q K_p^\top}{\sqrt{d_h}}\right).

    "Invalid channels are masked before the softmax operation. HbO and HbR
    share the same attention weights but retain separate value
    representations": :math:`Z^s_p = A_p W_V \mathrm{LN}(H^s_{:,p})`,
    :math:`s \in \{O, R\}`. "After an output projection and a feed-forward
    block", a chromophore-specific residual projection maps the feature
    dimension from :math:`D` to :math:`F`, giving
    :math:`Z^O, Z^R \in \mathbb{R}^{M \times P \times F}`.

    **Downstream adaptation.** The representations of both chromophores are
    "passed to a task-specific prediction head"; in the authors' benchmark,
    :math:`Z` "is flattened, standardized using training-set statistics,
    followed by dropout (:math:`p = 0.1`) and a single linear classification
    layer". Here ``head_dropout`` and ``final_layer`` are that head; the
    standardization is left to the caller.

    **Input.** Paired :math:`\Delta\text{HbO}`/:math:`\Delta\text{HbR}`
    channels, ``(batch, n_chans, n_times)``. The geometry comes from
    ``chs_info`` (source and detector in ``loc[3:9]``, metres) or the
    ``geometry`` argument of :meth:`encode` and ``forward``; as in the
    reference, a pair without source/detector locations gets a zero geometry
    and is left out of the montage centre. NaN samples are invalid; a patch
    with too few valid samples is left out of every attention.

    :meth:`load_state_dict` takes the official GeoNIRS-Base/Large checkpoints
    as released. The paper preprocesses every dataset the same way: optical
    density, quality control, the modified Beer-Lambert law (PPF 6.0, umol/L),
    resampling to 5 Hz, and "HbO and HbR are standardized separately for each
    recording" with "a single recording-level mean and standard deviation
    estimated from all valid channels and valid time points"; pretraining
    windows are 30 s (150 samples, 15 patches of 2 s). The 15-row temporal
    embedding of the checkpoints therefore loads only for ``n_times`` up to
    150.

    Parameters
    ----------
    patch_samples : int, default=10
        :math:`S`, samples per temporal patch (2 s at 5 Hz).
    max_patches : int, default=15
        Rows of the learnable temporal positional embedding, raised to the
        patch count of ``n_times``.
    on_non_divisible : {"error", "pad", "crop"}, default="error"
        A ``n_times`` that is not a multiple of ``patch_samples`` is refused,
        as by the reference, zero-padded or cropped.
    embedding_dim : int, default=192
        :math:`D`, width of the tokens (GeoNIRS-Large: 384).
    num_heads : int, default=8
        Heads of the temporal Transformers (GeoNIRS-Large: 12).
    temporal_layers : int, default=4
        Layers of each temporal Transformer (GeoNIRS-Large: 6).
    mlp_ratio : float, default=4.0
        Feed-forward width of the temporal Transformers, relative to
        ``embedding_dim``.
    geometry_hidden_dim : int or None, default=None
        Hidden width of the geometry MLP; ``None`` is ``embedding_dim``.
    geometry_heads : int or None, default=None
        Heads of the geometry Transformer layer; ``None`` is ``num_heads``.
    num_latent_sources : int, default=16
        :math:`M`, learnable latent queries.
    spatial_heads : int or None, default=None
        Heads of the geometry-aware channel attention (:math:`d_h` is
        ``embedding_dim // spatial_heads``); ``None`` is ``num_heads``.
    output_feature_dim : int, default=64
        :math:`F`, width of each latent token after the residual projection
        (GeoNIRS-Large: 128).
    use_channel_geometry : bool, default=True
        Add :math:`G` to the keys; ``False`` drops the geometry encoder (the
        paper's "w/o Geometry" variant).
    pool : {"flatten", "mean_std"}, default="flatten"
        Readout of the ``(batch, 2, M, P, F)`` representation: all of it, or
        the mean and standard deviation of :math:`[Z^O \| Z^R]` over the valid
        patches (the features of the reference ``mean_std_linear`` head, whose
        LayerNorm is left to the caller).
    channel_order : {"stacked", "interleaved"} or None, default=None
        Input layout, ``[HbO_1..HbO_C, HbR_1..HbR_C]`` or ``[HbO_1, HbR_1, ...]``;
        ``None`` reads it from ``chs_info``.
    drop_prob : float, default=0.1
        Dropout in the network.
    head_drop_prob : float, default=0.1
        Dropout before ``final_layer``.
    minimum_patch_valid_fraction : float, default=0.8
        A patch is valid when "at least 80% of its time points are valid".
    activation : type[nn.Module], default=nn.GELU
        Activation of the network.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info=None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        patch_samples: int = 10,
        max_patches: int = 15,
        on_non_divisible: Literal["error", "pad", "crop"] = "error",
        embedding_dim: int = 192,
        num_heads: int = 8,
        temporal_layers: int = 4,
        mlp_ratio: float = 4.0,
        geometry_hidden_dim: int | None = None,
        geometry_heads: int | None = None,
        num_latent_sources: int = 16,
        spatial_heads: int | None = None,
        output_feature_dim: int = 64,
        use_channel_geometry: bool = True,
        pool: Literal["flatten", "mean_std"] = "flatten",
        channel_order: Literal["stacked", "interleaved"] | None = None,
        drop_prob: float = 0.1,
        head_drop_prob: float = 0.1,
        minimum_patch_valid_fraction: float = 0.8,
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
        if self.n_chans % 2:
            raise ValueError("GeoNIRS needs HbO/HbR pairs: n_chans must be even.")
        self.channel_order = fnirs_channel_order(self._chs_info, channel_order)
        _check_on_non_divisible(on_non_divisible)
        self.on_non_divisible = on_non_divisible
        self.patch_samples = patch_samples
        self.n_patches = patch_count(self.n_times, patch_samples, on_non_divisible)
        self.embedding_dim = embedding_dim
        self.num_latent_sources = num_latent_sources
        self.pool = pool
        self.minimum_patch_valid_fraction = float(minimum_patch_valid_fraction)

        stream: tuple = (patch_samples, max(max_patches, self.n_patches), embedding_dim)
        stream += (num_heads, temporal_layers, mlp_ratio, drop_prob, activation)
        self.hbo_encoder = _TemporalStream(*stream)
        self.hbr_encoder = _TemporalStream(*stream)
        self.geometry_encoder = (
            _OptodeGeometryStream(
                embedding_dim,
                geometry_hidden_dim or embedding_dim,
                geometry_heads or num_heads,
                drop_prob,
                activation,
            )
            if use_channel_geometry
            else None
        )
        self.channel_compressor = _LatentSourceCrossAttention(
            embedding_dim,
            num_latent_sources,
            spatial_heads or num_heads,
            drop_prob,
            use_channel_geometry,
            activation,
        )
        self.hbo_projection = _cascade_projection(
            embedding_dim, output_feature_dim, drop_prob, activation
        )
        self.hbr_projection = _cascade_projection(
            embedding_dim, output_feature_dim, drop_prob, activation
        )
        self.register_buffer(
            "default_geometry",
            torch.from_numpy(_optode_geometry(self._chs_info, self.channel_order)),
            persistent=False,
        )
        self.feature_dim = {
            "flatten": 2 * num_latent_sources * self.n_patches * output_feature_dim,
            "mean_std": 4 * num_latent_sources * output_feature_dim,
        }[pool]
        self.head_dropout = nn.Dropout(head_drop_prob)
        self.final_layer = nn.Linear(self.feature_dim, self.n_outputs)

    def encode(
        self,
        x: torch.Tensor,
        geometry: torch.Tensor | None = None,
        ch_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """``(batch, n_chans, n_times)`` -> ``"foundation"``
        ``(batch, 2, S, P, output_feature_dim)`` and its validity ``"valid"``
        ``(batch, S, P)``. ``geometry`` is ``(n_pairs, 6)`` or
        ``(batch, n_pairs, 6)``; ``ch_mask`` ``(batch, n_pairs)`` flags real pairs.
        """
        x = pad_or_crop_to_patches(x, self.patch_samples, self.on_non_divisible)
        # (B, 2*C, T) -> 2 x (B, C, T)
        hbo, hbr = fnirs_split_hbo_hbr(x, self.channel_order)
        batch_size, n_pairs, _ = hbo.shape
        finite = torch.isfinite(hbo) & torch.isfinite(hbr)
        channel_valid = finite.any(dim=-1)  # (B, C)
        if ch_mask is not None:
            channel_valid = ch_mask.to(torch.bool) & channel_valid
        sample_valid = finite & channel_valid.unsqueeze(-1)  # (B, C, T)
        hbo = torch.where(sample_valid, hbo, torch.zeros_like(hbo))
        hbr = torch.where(sample_valid, hbr, torch.zeros_like(hbr))

        # (B, C, T) -> (B, C, P): valid with that fraction of valid samples
        ps = self.patch_samples
        n_valid = sample_valid.unfold(-1, ps, ps).sum(dim=-1)
        patch_valid = n_valid >= math.ceil(ps * self.minimum_patch_valid_fraction)
        patch_valid = patch_valid & channel_valid.unsqueeze(-1)

        if self.geometry_encoder is not None:
            geom = self.default_geometry if geometry is None else geometry
            if geom.numel() == 0:
                raise ValueError(
                    "GeoNIRS needs the channel geometry: chs_info with "
                    "source/detector locations (loc[3:9]) or the geometry argument."
                )
            if geometry is None and geom.shape[0] != n_pairs:
                raise ValueError(
                    "x and chs_info have different channel counts: pass the "
                    "geometry of the HbO/HbR pairs of x."
                )
            geom = geom.to(device=x.device, dtype=hbo.dtype)
            if geom.ndim == 2:
                geom = geom.unsqueeze(0).expand(batch_size, -1, -1)
            # (B, C, 6) -> (B, C, D)
            geometry_embedding = self.geometry_encoder(geom, channel_valid)
        else:
            geometry_embedding = hbo.new_zeros(batch_size, n_pairs, self.embedding_dim)
        # (B, C, T) -> (B, C, P, D) -> (B, S, P, D) -> (B, S, P, D_out)
        source_hbo, source_hbr = self.channel_compressor(
            self.hbo_encoder(hbo, patch_valid),
            self.hbr_encoder(hbr, patch_valid),
            geometry_embedding,
            patch_valid,
        )
        time_valid = patch_valid.any(dim=1)  # (B, P)
        valid_scale = time_valid[:, None, :, None].to(source_hbo.dtype)
        foundation = [
            self.hbo_projection(source_hbo) * valid_scale,
            self.hbr_projection(source_hbr) * valid_scale,
        ]
        return {
            "foundation": torch.stack(foundation, dim=1),
            "valid": time_valid[:, None, :].expand(
                batch_size, self.num_latent_sources, time_valid.shape[1]
            ),
        }

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.feature_dim, self.n_outputs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Also takes an official checkpoint (``{"model": ...}``); its
        pretraining decoders are dropped, a missing head keeps its init."""
        state_dict = state_dict.get("model", state_dict)
        decoders = ("hbo_to_hbr_decoder.", "hbr_to_hbo_decoder.")
        cleaned = {k: v for k, v in state_dict.items() if not k.startswith(decoders)}
        if not any(k.startswith("final_layer.") for k in cleaned):
            head = self.final_layer.state_dict(prefix="final_layer.")
            cleaned.update(head)
        return super().load_state_dict(cleaned, strict=strict, assign=assign)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        geometry: torch.Tensor | None = None,
        ch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """``geometry`` as in :meth:`encode`; ``ch_mask`` ``(batch, n_chans)``
        must pad whole HbO/HbR pairs, which every attention then leaves out."""
        ch_mask = check_ch_mask(ch_mask, x)
        if ch_mask is not None:  # (B, 2*C) -> (B, C)
            hbo_mask, hbr_mask = fnirs_split_hbo_hbr(ch_mask, self.channel_order)
            if not torch.equal(hbo_mask, hbr_mask):
                raise ValueError("ch_mask needs whole HbO/HbR pairs.")
            ch_mask = hbo_mask
        encoded = self.encode(x, geometry=geometry, ch_mask=ch_mask)
        foundation = encoded["foundation"]  # (B, 2, S, P, D_out)
        if self.pool == "flatten":
            features = foundation.flatten(1)
        else:  # (B, S, P, 2*D_out) -> (B, S * 4*D_out)
            joint = torch.cat([foundation[:, 0], foundation[:, 1]], dim=-1)
            mask = encoded["valid"].unsqueeze(-1).to(joint.dtype)
            count = mask.sum(dim=2).clamp_min(1.0)
            mean = (joint * mask).sum(dim=2) / count
            centred = (joint - mean.unsqueeze(2)) * mask
            std = (centred.square().sum(dim=2) / count).clamp_min(0.0).sqrt()
            features = torch.cat([mean, std], dim=-1).flatten(1)
        features = self.head_dropout(features)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)


class _TemporalStream(nn.Module):
    """Per-channel patch Transformer of one chromophore."""

    def __init__(self, ps, max_patches, dim, heads, layers, mlp_ratio, dropout, act):
        super().__init__()
        self.patch_samples = ps
        self.patch_projection = nn.Linear(ps, dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.time_embedding = nn.Embedding(max_patches, dim)
        ff_dim = round(dim * mlp_ratio)
        self.temporal_transformer = _transformer(
            dim, heads, ff_dim, dropout, act, layers
        )
        self.output_norm = nn.LayerNorm(dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.time_embedding.weight, std=0.02)

    def forward(self, signal: torch.Tensor, patch_valid: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, C, P, patch_samples) -> (B, C, P, D)
        patches = signal.unfold(-1, self.patch_samples, self.patch_samples)
        tokens = self.patch_norm(self.patch_projection(patches))
        valid = patch_valid.unsqueeze(-1).to(tokens.dtype)
        tokens = (tokens + self.time_embedding.weight[: patches.shape[2]]) * valid
        # (B*C, P, D); a channel with no valid patch attends to all, then is zeroed
        rows = patch_valid.flatten(0, 1)
        safe_valid = rows | (~rows.any(dim=1, keepdim=True))
        encoded = self.temporal_transformer(
            tokens.flatten(0, 1), src_key_padding_mask=~safe_valid
        )
        return self.output_norm(encoded.reshape(tokens.shape) * valid) * valid


class _OptodeGeometryStream(nn.Module):
    """MLP and one spatial Transformer layer over the 6D channel geometry."""

    def __init__(self, dim, hidden_dim, heads, dropout, activation):
        super().__init__()
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )
        self.spatial_transformer = _transformer(
            dim, heads, dim * 2, dropout, activation, 1
        )

    def forward(self, geometry: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid_f = valid.unsqueeze(-1)  # (B, C, 1)
        geometry = torch.where(valid_f, geometry, torch.zeros_like(geometry))
        tokens = self.geometry_projection(geometry)  # (B, C, 6) -> (B, C, D)
        tokens = tokens * valid_f.to(tokens.dtype)
        safe_valid = valid | (~valid.any(dim=1, keepdim=True))
        encoded = self.spatial_transformer(tokens, src_key_padding_mask=~safe_valid)
        return encoded * valid_f.to(encoded.dtype)


class _LatentSourceCrossAttention(nn.Module):
    """The authors' ``BrainOmniChannelCompressor``: cross-attention of
    ``num_latents`` learned queries over the channels.

    Private because one softmax over the fused HbO/HbR keys weighs both
    chromophores' values (one dropout mask), and the released checkpoints
    name its projections ``q_proj``/``k_proj``/``v_proj``/``out_proj``.
    """

    def __init__(self, dim, num_latents, num_heads, dropout, use_geometry, activation):
        super().__init__()
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_channel_geometry = use_geometry
        self.latent_queries = nn.Parameter(torch.empty(num_latents, dim))
        self.q_norm, self.k_norm, self.v_norm = (nn.LayerNorm(dim) for _ in range(3))
        projections = (nn.Linear(dim, dim) for _ in range(4))
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = projections
        self.attn_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.latent_queries, 0.0, 0.02)

    def forward(
        self,
        temporal_hbo: torch.Tensor,
        temporal_hbr: torch.Tensor,
        geometry_embedding: torch.Tensor,
        patch_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_chans, n_patches, dim = temporal_hbo.shape
        bp = batch_size * n_patches
        # (B, C, P, D) -> (B*P, C, D): attention over the channels of each patch
        hbo = temporal_hbo.permute(0, 2, 1, 3).reshape(bp, n_chans, dim)
        hbr = temporal_hbr.permute(0, 2, 1, 3).reshape(bp, n_chans, dim)
        valid = patch_valid.permute(0, 2, 1).reshape(bp, n_chans)
        geometry = (
            geometry_embedding.unsqueeze(1)
            .expand(batch_size, n_patches, n_chans, dim)
            .reshape(bp, n_chans, dim)
        )
        q_input = self.q_norm(self.latent_queries).unsqueeze(0).expand(bp, -1, -1)
        fused = 0.5 * (hbo + hbr)
        k_in = fused + geometry if self.use_channel_geometry else fused
        # (B*P, S or C, D) -> (B*P, heads, S or C, head_dim)
        heads = [self.num_heads, self.head_dim]
        q = self.q_proj(q_input).unflatten(-1, heads).transpose(1, 2)
        k = self.k_proj(self.k_norm(k_in)).unflatten(-1, heads).transpose(1, 2)
        # a patch with no valid channel attends to all of them, then is zeroed
        active = valid.any(dim=1, keepdim=True)  # (B*P, 1)
        # shared attention weights over [V_hbo | V_hbr]: (B*P, h, C, 2*head_dim)
        v = torch.cat(
            [
                self.v_proj(self.v_norm(s)).unflatten(-1, heads).transpose(1, 2)
                for s in [hbo, hbr]
            ],
            dim=-1,
        )
        attended = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=(valid | ~active)[:, None, None, :],
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=self.head_dim**-0.5,
            exact=True,
        )  # (B*P, h, S, 2*head_dim)
        outputs: List[torch.Tensor] = []
        for x in attended.chunk(2, dim=-1):
            x = x.transpose(1, 2).reshape(bp, self.num_latents, dim)
            x = self.out_norm(q_input + self.out_proj(x))
            x = (x + self.ffn(x)) * active.unsqueeze(-1).to(x.dtype)
            # (B*P, S, D) -> (B, S, P, D)
            outputs.append(
                x.reshape(batch_size, n_patches, self.num_latents, dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
        return outputs[0], outputs[1]


class _ResidualProjectionBlock(nn.Module):
    """The authors' ``ResidualProjectionBlock``: ``skip(x) + residual_scale *
    main(x)``, ``residual_scale`` starting at 0.1.

    Private because no shared block has a learned residual scale on a
    width-changing skip projection.
    """

    def __init__(self, input_dim, output_dim, dropout, activation):
        super().__init__()
        self.skip = nn.Linear(input_dim, output_dim)
        self.main = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.residual_scale = nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.residual_scale, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.residual_scale * self.main(x)


def _transformer(dim, heads, ff_dim, dropout, activation, n_layers):
    """Pre-norm ``nn.TransformerEncoder`` with a final LayerNorm."""
    kwargs = dict(batch_first=True, norm_first=True)
    layer = nn.TransformerEncoderLayer(
        dim, heads, ff_dim, dropout, activation(), **kwargs
    )
    return nn.TransformerEncoder(
        layer, n_layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False
    )


def _cascade_projection(input_dim, output_dim, dropout, activation):
    """Three residual stages from ``input_dim`` to ``output_dim``."""
    # GeoNIRS-Base was pretrained with 192 -> 128 -> 96 -> 64
    if input_dim == 192 and output_dim == 64:
        widths = [192, 128, 96, 64]
    else:
        widths = [input_dim, input_dim // 2, input_dim // 4, output_dim]
    stages = [
        _ResidualProjectionBlock(widths[i], widths[i + 1], dropout, activation)
        for i in range(3)
    ]
    return nn.Sequential(OrderedDict(blocks=nn.Sequential(*stages)))


def _optode_geometry(chs_info, channel_order: str) -> np.ndarray:
    """``(n_pairs, 6)`` ``[midpoint, detector - source]`` of the HbO channels,
    centred on the mean optode and divided by the 25.46 mm of the GeoNIRS
    pretraining. As the reference (``scripts/build_fair_input.py``), a pair
    without source/detector locations is zero and left out of the centre;
    ``(0, 6)`` when no pair has them."""
    empty = np.zeros((0, 6), dtype=np.float32)
    if chs_info is None:
        return empty
    hbo_index, _ = fnirs_split_hbo_hbr(torch.arange(len(chs_info))[None], channel_order)
    # (n_pairs, 6): source and detector, loc[3:9]; a missing or short loc stays zero
    optodes = np.zeros((hbo_index.shape[1], 6))
    for row, i in zip(optodes, hbo_index[0].tolist()):
        loc = chs_info[i].get("loc")
        loc = np.ravel(loc if loc is not None else [])[3:9]
        row[: loc.size] = loc
    present = np.isfinite(optodes).all(axis=1) & (optodes != 0).any(axis=1)
    if not present.any():
        return empty
    scale = 25.460582556839242 / 1000.0
    center = np.concatenate([optodes[present, :3], optodes[present, 3:]]).mean(axis=0)
    src, det = (optodes[:, :3] - center) / scale, (optodes[:, 3:] - center) / scale
    geometry = np.concatenate([0.5 * (src + det), det - src], axis=1)
    geometry[~present] = 0.0
    return geometry.astype(np.float32)
