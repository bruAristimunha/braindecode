# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3-Clause
# Adapted from https://github.com/GeoNIRS/GeoNIRS (81f8995; no licence file;
# paper arXiv:2610.17845).

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Literal, Optional

import numpy as np
import torch
from einops.layers.torch import Rearrange
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
        if pool not in ("flatten", "mean_std"):
            raise ValueError(f"pool must be 'flatten' or 'mean_std', got {pool!r}.")
        self.channel_order = fnirs_channel_order(self._chs_info, channel_order)
        _check_on_non_divisible(on_non_divisible)
        self.on_non_divisible = on_non_divisible
        self.patch_samples = patch_samples
        self.n_patches = patch_count(self.n_times, patch_samples, on_non_divisible)
        self.min_valid_samples = math.ceil(
            patch_samples * float(minimum_patch_valid_fraction)
        )
        self.num_latent_sources = num_latent_sources
        self.pool = pool

        temporal_encoder = dict(
            patch_samples=patch_samples,
            max_patches=max(max_patches, self.n_patches),
            dim=embedding_dim,
            num_heads=num_heads,
            num_layers=temporal_layers,
            mlp_ratio=mlp_ratio,
            dropout=drop_prob,
            activation=activation,
        )
        self.hbo_encoder = _TemporalEncoder(**temporal_encoder)
        self.hbr_encoder = _TemporalEncoder(**temporal_encoder)
        self.geometry_encoder = None
        if use_channel_geometry:
            self.geometry_encoder = _GeometryEncoder(
                dim=embedding_dim,
                hidden_dim=geometry_hidden_dim or embedding_dim,
                num_heads=geometry_heads or num_heads,
                dropout=drop_prob,
                activation=activation,
            )
        self.channel_compressor = _LatentSourceCrossAttention(
            dim=embedding_dim,
            num_latents=num_latent_sources,
            num_heads=spatial_heads or num_heads,
            dropout=drop_prob,
            activation=activation,
        )
        self.hbo_projection = _residual_projection(
            embedding_dim, output_feature_dim, drop_prob, activation
        )
        self.hbr_projection = _residual_projection(
            embedding_dim, output_feature_dim, drop_prob, activation
        )
        self.register_buffer(
            "default_geometry",
            torch.from_numpy(_optode_geometry(self._chs_info, self.channel_order)),
            persistent=False,
        )
        self.split_patches = Rearrange(
            "batch pairs (patches samples) -> batch pairs patches samples",
            samples=patch_samples,
        )
        self.pair_over_time = Rearrange("batch pairs -> batch pairs 1")
        self.patch_over_sources = Rearrange("batch patches -> batch 1 patches")
        self.patch_over_features = Rearrange("batch patches -> batch 1 patches 1")
        self.flatten_foundation = Rearrange(
            "batch chrom sources patches dim -> batch (chrom sources patches dim)"
        )
        self.join_chromophores = Rearrange(
            "batch chrom sources patches dim -> batch sources patches (chrom dim)"
        )
        self.valid_over_features = Rearrange(
            "batch sources patches -> batch sources patches 1"
        )
        self.mean_over_patches = Rearrange("batch sources dim -> batch sources 1 dim")
        self.flatten_statistics = Rearrange(
            "stat batch sources dim -> batch (sources stat dim)"
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
        geometry: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """``(batch, n_chans, n_times)`` -> ``"foundation"``
        ``(batch, 2, M, P, output_feature_dim)`` (:math:`Z^O, Z^R`) and its
        validity ``"valid"`` ``(batch, M, P)``. ``geometry`` is
        ``(n_pairs, 6)`` or ``(batch, n_pairs, 6)``; ``ch_mask``
        ``(batch, n_pairs)`` flags real pairs.
        """
        # (batch, n_chans, n_times) -> (batch, n_chans, P * S)
        x = pad_or_crop_to_patches(x, self.patch_samples, self.on_non_divisible)
        # (batch, n_chans, P * S) -> 2 x (batch, n_pairs, P * S)
        hbo, hbr = fnirs_split_hbo_hbr(x, self.channel_order)
        batch_size, n_pairs, _ = hbo.shape
        hbo_finite = torch.isfinite(hbo)
        hbr_finite = torch.isfinite(hbr)
        # (batch, n_pairs, P * S): both chromophores finite
        finite = hbo_finite & hbr_finite
        # (batch, n_pairs, P * S) -> (batch, n_pairs): a pair with a finite sample
        channel_valid = finite.any(dim=-1)
        if ch_mask is not None:
            ch_mask = ch_mask.to(torch.bool)
            channel_valid = ch_mask & channel_valid
        # (batch, n_pairs) -> (batch, n_pairs, 1)
        channel_valid_over_time = self.pair_over_time(channel_valid)
        # (batch, n_pairs, P * S)
        sample_valid = finite & channel_valid_over_time
        zeros = torch.zeros_like(hbo)
        hbo = torch.where(sample_valid, hbo, zeros)
        hbr = torch.where(sample_valid, hbr, zeros)

        # (batch, n_pairs, P * S) -> (batch, n_pairs, P, S)
        sample_valid_patches = self.split_patches(sample_valid)
        # (batch, n_pairs, P, S) -> (batch, n_pairs, P)
        n_valid_samples = sample_valid_patches.sum(dim=-1)
        patch_valid = n_valid_samples >= self.min_valid_samples
        patch_valid = patch_valid & channel_valid_over_time

        geometry_embedding: Optional[torch.Tensor] = None
        if self.geometry_encoder is not None:
            if geometry is None:
                pair_geometry = self.default_geometry
                if pair_geometry.numel() == 0:
                    raise ValueError(
                        "GeoNIRS needs the channel geometry: chs_info with "
                        "source/detector locations (loc[3:9]) or the geometry "
                        "argument."
                    )
                if pair_geometry.shape[0] != n_pairs:
                    raise ValueError(
                        "x and chs_info have different channel counts: pass the "
                        "geometry of the HbO/HbR pairs of x."
                    )
            else:
                pair_geometry = geometry
                if pair_geometry.shape[-2] != n_pairs:
                    raise ValueError(
                        "geometry and x have different channel counts: pass one "
                        "row per HbO/HbR pair of x."
                    )
            pair_geometry = pair_geometry.to(device=x.device, dtype=hbo.dtype)
            # (n_pairs, 6) or (batch, n_pairs, 6) -> (batch, n_pairs, 6)
            pair_geometry = pair_geometry.expand(batch_size, -1, -1)
            # (batch, n_pairs, 6) -> (batch, n_pairs, D): G
            geometry_embedding = self.geometry_encoder(pair_geometry, channel_valid)
        # (batch, n_pairs, P * S) -> (batch, n_pairs, P, D): H^O, H^R
        temporal_hbo = self.hbo_encoder(hbo, patch_valid)
        temporal_hbr = self.hbr_encoder(hbr, patch_valid)
        # -> 2 x (batch, M, P, D)
        source_hbo, source_hbr = self.channel_compressor(
            temporal_hbo, temporal_hbr, geometry_embedding, patch_valid
        )
        # (batch, n_pairs, P) -> (batch, P): a patch with a valid pair
        time_valid = patch_valid.any(dim=1)
        # (batch, P) -> (batch, 1, P, 1)
        valid_scale = self.patch_over_features(time_valid)
        valid_scale = valid_scale.to(source_hbo.dtype)
        # (batch, M, P, D) -> (batch, M, P, F): Z^O, Z^R
        compact_hbo = self.hbo_projection(source_hbo)
        compact_hbo = compact_hbo * valid_scale
        compact_hbr = self.hbr_projection(source_hbr)
        compact_hbr = compact_hbr * valid_scale
        # 2 x (batch, M, P, F) -> (batch, 2, M, P, F)
        foundation = torch.stack([compact_hbo, compact_hbr], dim=1)
        # (batch, P) -> (batch, 1, P) -> (batch, M, P)
        valid = self.patch_over_sources(time_valid)
        valid = valid.expand(-1, self.num_latent_sources, -1)
        return {"foundation": foundation, "valid": valid}

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

    def _valid_mean_std(
        self, foundation: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Mean and standard deviation of :math:`[Z^O \\| Z^R]` over the valid
        patches, as the reference ``valid_temporal_mean_std``."""
        # (batch, 2, M, P, F) -> (batch, M, P, 2 F)
        joint = self.join_chromophores(foundation)
        # (batch, M, P) -> (batch, M, P, 1)
        weight = self.valid_over_features(valid)
        weight = weight.to(joint.dtype)
        # (batch, M, P, 1) -> (batch, M, 1)
        count = weight.sum(dim=2)
        count = count.clamp_min(1.0)
        # (batch, M, P, 2 F) -> (batch, M, 2 F)
        weighted = joint * weight
        total = weighted.sum(dim=2)
        mean = total / count
        # (batch, M, 2 F) -> (batch, M, 1, 2 F)
        mean_over_patches = self.mean_over_patches(mean)
        centred = joint - mean_over_patches
        centred = centred * weight
        squared = centred.square()
        # (batch, M, P, 2 F) -> (batch, M, 2 F)
        variance = squared.sum(dim=2)
        variance = variance / count
        variance = variance.clamp_min(0.0)
        std = variance.sqrt()
        # 2 x (batch, M, 2 F) -> (2, batch, M, 2 F)
        statistics = torch.stack([mean, std])
        # (2, batch, M, 2 F) -> (batch, M * 4 F)
        return self.flatten_statistics(statistics)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        geometry: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """``geometry`` as in :meth:`encode`; ``ch_mask`` ``(batch, n_chans)``
        must pad whole HbO/HbR pairs, which every attention then leaves out."""
        ch_mask = check_ch_mask(ch_mask, x)
        if ch_mask is not None:
            # (batch, n_chans) -> 2 x (batch, n_pairs)
            hbo_mask, hbr_mask = fnirs_split_hbo_hbr(ch_mask, self.channel_order)
            if not torch.equal(hbo_mask, hbr_mask):
                raise ValueError("ch_mask needs whole HbO/HbR pairs.")
            ch_mask = hbo_mask
        encoded = self.encode(x, geometry=geometry, ch_mask=ch_mask)
        # (batch, 2, M, P, F)
        foundation = encoded["foundation"]
        if self.pool == "flatten":
            # (batch, 2, M, P, F) -> (batch, 2 * M * P * F)
            features = self.flatten_foundation(foundation)
        else:
            # (batch, 2, M, P, F) -> (batch, M * 4 F)
            features = self._valid_mean_std(foundation, encoded["valid"])
        features = self.head_dropout(features)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)


class _TemporalEncoder(nn.Module):
    """Temporal encoder of one chromophore: patch projection, LayerNorm and a
    learnable temporal positional embedding, then a Transformer over the
    patches of each channel (the authors' ``TemporalStreamEncoder``)."""

    def __init__(
        self,
        patch_samples,
        max_patches,
        dim,
        num_heads,
        num_layers,
        mlp_ratio,
        dropout,
        activation,
    ):
        super().__init__()
        self.split_patches = Rearrange(
            "batch pairs (patches samples) -> batch pairs patches samples",
            samples=patch_samples,
        )
        self.patch_projection = nn.Linear(patch_samples, dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.time_embedding = nn.Embedding(max_patches, dim)
        self.temporal_transformer = _transformer(
            dim, num_heads, round(dim * mlp_ratio), dropout, activation, num_layers
        )
        self.output_norm = nn.LayerNorm(dim)
        self.patch_over_features = Rearrange(
            "batch pairs patches -> batch pairs patches 1"
        )
        self.merge_pairs = Rearrange(
            "batch pairs patches dim -> (batch pairs) patches dim"
        )
        self.merge_pair_mask = Rearrange("batch pairs patches -> (batch pairs) patches")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.time_embedding.weight, std=0.02)

    def forward(self, signal: torch.Tensor, patch_valid: torch.Tensor) -> torch.Tensor:
        # (batch, n_pairs, P * S) -> (batch, n_pairs, P, S)
        patches = self.split_patches(signal)
        # (batch, n_pairs, P, S) -> (batch, n_pairs, P, D)
        tokens = self.patch_projection(patches)
        tokens = self.patch_norm(tokens)
        # (P, D)
        time_position = self.time_embedding.weight[: patches.shape[2]]
        tokens = tokens + time_position
        # (batch, n_pairs, P) -> (batch, n_pairs, P, 1)
        weight = self.patch_over_features(patch_valid)
        weight = weight.to(tokens.dtype)
        tokens = tokens * weight
        # (batch, n_pairs, P) -> (batch * n_pairs, P)
        rows_valid = self.merge_pair_mask(patch_valid)
        # a channel with no valid patch attends to all of them, then is zeroed
        row_active = rows_valid.any(dim=1, keepdim=True)
        attend = rows_valid | ~row_active
        padding = ~attend
        # (batch, n_pairs, P, D) -> (batch * n_pairs, P, D)
        sequences = self.merge_pairs(tokens)
        encoded = self.temporal_transformer(sequences, src_key_padding_mask=padding)
        # (batch * n_pairs, P, D) -> (batch, n_pairs, P, D)
        encoded = encoded.reshape(tokens.shape)
        encoded = encoded * weight
        encoded = self.output_norm(encoded)
        return encoded * weight


class _GeometryEncoder(nn.Module):
    """Geometry encoder: an MLP and one Transformer layer over the 6D
    source-detector geometry of the channels (the authors'
    ``SourceDetectorGeometryEncoder``)."""

    def __init__(self, dim, hidden_dim, num_heads, dropout, activation):
        super().__init__()
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )
        self.spatial_transformer = _transformer(
            dim, num_heads, dim * 2, dropout, activation, 1
        )
        self.pair_over_features = Rearrange("batch pairs -> batch pairs 1")

    def forward(self, geometry: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # (batch, n_pairs) -> (batch, n_pairs, 1)
        valid_over_features = self.pair_over_features(valid)
        zeros = torch.zeros_like(geometry)
        # (batch, n_pairs, 6): an invalid pair has no geometry
        geometry = torch.where(valid_over_features, geometry, zeros)
        # (batch, n_pairs, 6) -> (batch, n_pairs, D)
        tokens = self.geometry_projection(geometry)
        weight = valid_over_features.to(tokens.dtype)
        tokens = tokens * weight
        # a sample with no valid pair attends to all of them, then is zeroed
        any_valid = valid.any(dim=1, keepdim=True)
        attend = valid | ~any_valid
        padding = ~attend
        encoded = self.spatial_transformer(tokens, src_key_padding_mask=padding)
        return encoded * weight


class _LatentSourceCrossAttention(nn.Module):
    """The geometry-aware channel compressor (the authors'
    ``BrainOmniChannelCompressor``): cross-attention of ``num_latents``
    learned queries over the channels, then an output projection and a
    feed-forward block.

    Private because one softmax over the fused HbO/HbR keys weighs both
    chromophores' values (one dropout mask), and the released checkpoints
    name its projections ``q_proj``/``k_proj``/``v_proj``/``out_proj``.
    """

    def __init__(self, dim, num_latents, num_heads, dropout, activation):
        super().__init__()
        self.head_dim = dim // num_heads
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
        self.patches_first = Rearrange(
            "batch pairs patches dim -> (batch patches) pairs dim"
        )
        self.patches_first_mask = Rearrange(
            "batch pairs patches -> (batch patches) pairs"
        )
        self.geometry_over_patches = Rearrange("batch pairs dim -> batch 1 pairs dim")
        self.merge_patches = Rearrange(
            "batch patches pairs dim -> (batch patches) pairs dim"
        )
        self.latents_over_batch = Rearrange("latents dim -> 1 latents dim")
        self.split_heads = Rearrange(
            "batch tokens (heads dim) -> batch heads tokens dim", heads=num_heads
        )
        self.keys_over_queries = Rearrange("batch pairs -> batch 1 1 pairs")
        self.merge_heads = Rearrange(
            "batch heads latents (chrom dim) -> chrom batch latents (heads dim)",
            chrom=2,
        )
        self.active_over_features = Rearrange("batch one -> batch one 1")
        self.sources_first = Rearrange(
            "batch patches latents dim -> batch latents patches dim"
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.latent_queries, 0.0, 0.02)

    def forward(
        self,
        temporal_hbo: torch.Tensor,
        temporal_hbr: torch.Tensor,
        geometry_embedding: Optional[torch.Tensor],
        patch_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, n_patches, _ = temporal_hbo.shape
        # (batch, n_pairs, P, D) -> (batch * P, n_pairs, D): the channels of each patch
        hbo = self.patches_first(temporal_hbo)
        hbr = self.patches_first(temporal_hbr)
        # (batch, n_pairs, P) -> (batch * P, n_pairs)
        valid = self.patches_first_mask(patch_valid)
        # (M, D) -> (1, M, D) -> (batch * P, M, D): LN(L)
        latents = self.q_norm(self.latent_queries)
        latents = self.latents_over_batch(latents)
        latents = latents.expand(valid.shape[0], -1, -1)
        # (batch * P, n_pairs, D): fused HbO/HbR
        fused = hbo + hbr
        fused = 0.5 * fused
        keys_input = fused
        if geometry_embedding is not None:
            # (batch, n_pairs, D) -> (batch, 1, n_pairs, D) -> (batch * P, n_pairs, D)
            geometry = self.geometry_over_patches(geometry_embedding)
            geometry = geometry.expand(-1, n_patches, -1, -1)
            geometry = self.merge_patches(geometry)
            keys_input = fused + geometry
        # (batch * P, M, D) -> (batch * P, heads, M, d_h): Q = W_Q LN(L)
        query = self.q_proj(latents)
        query = self.split_heads(query)
        # (batch * P, n_pairs, D) -> (batch * P, heads, n_pairs, d_h): K = W_K LN(.)
        keys_input = self.k_norm(keys_input)
        key = self.k_proj(keys_input)
        key = self.split_heads(key)
        # (batch * P, n_pairs, D) -> (batch * P, heads, n_pairs, d_h): W_V LN(H^s)
        value_hbo = self.v_norm(hbo)
        value_hbo = self.v_proj(value_hbo)
        value_hbo = self.split_heads(value_hbo)
        value_hbr = self.v_norm(hbr)
        value_hbr = self.v_proj(value_hbr)
        value_hbr = self.split_heads(value_hbr)
        # shared attention weights over [V^O | V^R]: (batch * P, heads, n_pairs, 2 d_h)
        value = torch.cat([value_hbo, value_hbr], dim=-1)
        # (batch * P, 1): a patch with no valid channel attends to all, then is zeroed
        active = valid.any(dim=1, keepdim=True)
        attend = valid | ~active
        # (batch * P, n_pairs) -> (batch * P, 1, 1, n_pairs)
        attend = self.keys_over_queries(attend)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        # -> (batch * P, heads, M, 2 d_h)
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attend,
            dropout_p=dropout_p,
            scale=self.head_dim**-0.5,
            exact=True,
        )
        # (batch * P, heads, M, 2 d_h) -> (2, batch * P, M, D): Z^O, Z^R
        compressed = self.merge_heads(attended)
        # (batch * P, 1) -> (batch * P, 1, 1)
        active_weight = self.active_over_features(active)
        active_weight = active_weight.to(compressed.dtype)
        source_hbo = self._output_block(
            compressed[0], latents, active_weight, batch_size, n_patches
        )
        source_hbr = self._output_block(
            compressed[1], latents, active_weight, batch_size, n_patches
        )
        return source_hbo, source_hbr

    def _output_block(
        self,
        compressed: torch.Tensor,
        latents: torch.Tensor,
        active_weight: torch.Tensor,
        batch_size: int,
        n_patches: int,
    ) -> torch.Tensor:
        # (batch * P, M, D): output projection, residual on LN(L), LayerNorm
        projected = self.out_proj(compressed)
        hidden = latents + projected
        hidden = self.out_norm(hidden)
        # feed-forward block
        update = self.ffn(hidden)
        hidden = hidden + update
        hidden = hidden * active_weight
        # (batch * P, M, D) -> (batch, P, M, D)
        hidden = hidden.unflatten(0, [batch_size, n_patches])
        # (batch, P, M, D) -> (batch, M, P, D)
        hidden = self.sources_first(hidden)
        return hidden.contiguous()


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
        skip = self.skip(x)
        update = self.main(x)
        update = self.residual_scale * update
        return skip + update


def _transformer(dim, heads, ff_dim, dropout, activation, n_layers):
    """Pre-norm ``nn.TransformerEncoder`` with a final LayerNorm."""
    layer = nn.TransformerEncoderLayer(
        dim, heads, ff_dim, dropout, activation(), batch_first=True, norm_first=True
    )
    return nn.TransformerEncoder(
        layer, n_layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False
    )


def _residual_projection(input_dim, output_dim, dropout, activation):
    """Chromophore-specific residual projection from :math:`D` to :math:`F`:
    three residual stages, ``[192, 128, 96, 64]`` for GeoNIRS-Base and
    ``[D, D / 2, D / 4, F]`` otherwise (``[384, 192, 96, 128]`` for Large),
    as the paper's Table 6."""
    widths = [input_dim, input_dim // 2, input_dim // 4, output_dim]
    if input_dim == 192 and output_dim == 64:
        widths = [192, 128, 96, 64]
    stages = [
        _ResidualProjectionBlock(widths[i], widths[i + 1], dropout, activation)
        for i in range(3)
    ]
    # ``blocks`` is the checkpoints' key path
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
    channel_index = torch.arange(len(chs_info))[None]
    hbo_index, _ = fnirs_split_hbo_hbr(channel_index, channel_order)
    # (n_pairs, 6): source and detector, loc[3:9]; a missing or short loc stays zero
    optodes = np.zeros((hbo_index.shape[1], 6))
    for row, i in zip(optodes, hbo_index[0].tolist()):
        loc = chs_info[i].get("loc")
        loc = np.ravel(loc if loc is not None else [])[3:9]
        row[: loc.size] = loc
    present = np.isfinite(optodes).all(axis=1) & (optodes != 0).any(axis=1)
    if not present.any():
        return empty
    scale = 25.460582556839242 / 1000.0  # the 25.46 mm reference scale, in metres
    present_optodes = np.concatenate([optodes[present, :3], optodes[present, 3:]])
    center = present_optodes.mean(axis=0)
    source = (optodes[:, :3] - center) / scale
    detector = (optodes[:, 3:] - center) / scale
    geometry = np.concatenate([0.5 * (source + detector), detector - source], axis=1)
    geometry[~present] = 0.0
    return geometry.astype(np.float32)
