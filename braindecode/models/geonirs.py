# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# Adapted from https://github.com/GeoNIRS/GeoNIRS (81f8995; paper arXiv:2610.17845).

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
    r"""GeoNIRS from Li et al. (2026).

    :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    GeoNIRS is "a geometry-aware pretraining framework that uses the physical
    locations of source-detector pairs as a shared spatial reference across
    heterogeneous probe layouts" :footcite:`li2026geonirs`. Separate HbO and
    HbR temporal Transformers encode patches of :math:`S` samples of each
    channel. A geometry encoder embeds each pair's midpoint and
    source-detector vector, and :math:`M` learnable latent queries attend over
    the channels (the keys carry the geometry; HbO and HbR share the attention
    weights), compressing any montage into a fixed
    :math:`2 \times M \times P \times F` representation. This class is that
    encoder with a linear head, without the pretraining decoders.

    .. rubric:: How braindecode maps it

    - **Input.** ``(batch, 2C, n_times)`` paired HbO/HbR channels, stacked or
      interleaved (the MNE layout); ``channel_order=None`` reads the layout
      from ``chs_info``. Non-finite samples are invalid; invalid pairs and
      patches are left out of every attention, and ``ch_mask`` marks padding.
    - **Geometry.** Source and detector positions from ``chs_info``
      (``loc[3:9]``), centred on the montage and divided by 25.46 mm, as in
      the paper's downstream evaluation. The ``geometry`` argument of
      :meth:`encode` and ``forward`` replaces them.
    - **Left out:** the pretraining decoders, the head's training-set feature
      standardization (left to the caller) and LoRA. **Added:**
      ``pool="mean_std"``, ``on_non_divisible="pad"``/``"crop"`` and ``ch_mask``.
    - **Names.** Arguments take braindecode's names; each parameter below
      gives the authors' ``model_config`` key it replaces. The state-dict keys
      are the checkpoints' own, so they load with ``strict=True`` and no
      ``mapping``.

    .. rubric:: Variants

    Both released backbones use :math:`M = 16` and :math:`S = 10` (paper
    Table 6); the defaults build GeoNIRS-Base.

    =====================================  =====================  ======================
    Component                              GeoNIRS-Base           GeoNIRS-Large
    =====================================  =====================  ======================
    Pretraining corpus                     8 datasets, 383.30 h   22 datasets, 827.04 h
    ``embed_dim`` (:math:`D`)              192                    384
    ``depth`` / ``num_heads``              4 / 8                  6 / 12
    ``feature_dim`` (:math:`F`)            64                     128
    Parameters, checkpoint with decoders   6,511,214              30,146,414
    Parameters, this class (no head)       4,443,462              24,394,310
    Released checkpoint                    ``geonirs_base.pt``    ``geonirs_large.pt``
    =====================================  =====================  ======================

    Parameters
    ----------
    patch_size : int, default=10
        :math:`S`, samples per patch: 2 s at 5 Hz, as in both checkpoints
        (the reference's ``patch_samples``).
    max_patches : int, default=15
        Rows of the temporal positional embedding (30 s pretraining windows),
        raised to the patch count of ``n_times``; 15 loads the checkpoints.
    on_non_divisible : {"error", "pad", "crop"}, default="error"
        For ``n_times`` not a multiple of ``patch_size``: refuse, as the
        reference does, zero-pad or crop.
    embed_dim : int, default=192
        :math:`D`, token width (the reference's ``embedding_dim``).
    num_heads : int, default=8
        Heads of the temporal Transformers.
    depth : int, default=4
        Layers of each temporal Transformer (the reference's
        ``temporal_layers``).
    mlp_ratio : float, default=4.0
        Feed-forward width of the temporal Transformers, relative to :math:`D`.
    geometry_hidden_dim : int or None, default=None
        Hidden width of the geometry MLP; ``None`` is ``embed_dim``.
    geometry_heads : int or None, default=None
        Heads of the geometry Transformer layer; ``None`` is ``num_heads``.
    n_latent_channels : int, default=16
        :math:`M`, learnable latent queries, the channels of the compressed
        montage (the reference's ``num_latent_sources``).
    compressor_heads : int or None, default=None
        Heads of the channel compressor's attention; ``None`` is
        ``num_heads`` (the reference's ``spatial_heads``).
    feature_dim : int, default=64
        :math:`F`, width of each latent token after the chromophore projection
        (the reference's ``output_feature_dim``).
    use_channel_geometry : bool, default=True
        Add the geometry embedding to the keys; ``False`` drops that encoder.
    pool : {"flatten", "mean_std"}, default="flatten"
        Readout of the ``(batch, 2, M, P, F)`` tokens: ``"flatten"``, all
        tokens flattened (the paper's ``flatten_linear`` head); ``"mean_std"``,
        their mean and standard deviation over the valid patches
        (``mean_std_linear``).
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_C, HbR_1..HbR_C]`` or ``[HbO_1, HbR_1, ...]``; ``None``
        reads it from ``chs_info`` (stacked without it).
    drop_prob : float, default=0.1
        Dropout in the network (the reference's ``dropout``).
    head_drop_prob : float, default=0.1
        Dropout before ``final_layer``, the paper's :math:`p = 0.1`.
    minimum_patch_valid_fraction : float, default=0.8
        A patch is valid when at least this fraction of its samples is finite.
    activation : type[nn.Module], default=nn.GELU
        Activation of the network, as in the reference.

    Notes
    -----
    **Protocol** (paper Appendix B, Tables 7-9). Pretraining: optical density,
    quality control, modified Beer-Lambert law (PPF 6.0), 5 Hz, HbO and HbR
    z-scored separately per recording, no band-pass, 30 s windows; AdamW with
    cosine decay, Base at 1e-4 for 150 epochs, Large at 3e-4 for 40.
    Downstream: eight held-out binary tasks, a fixed channel set per dataset,
    12 s windows (60 samples) for six tasks and 10 s for two; subject-level
    five-fold cross-validation with seeds 810 and 811 (mean of 10 cells);
    AdamW, 30 epochs, batch 16, class-weighted cross-entropy, selection on
    validation balanced accuracy; linear probing trains the head at 1e-4.
    Balanced accuracy is the main metric, macro-F1 and AUROC the others.

    **Replication.** Linear probe of the frozen GeoNIRS-Base on DS42 (action
    mode, ds005935) and DS19 (pain intensity, ds005777), 60-sample windows,
    folds 1-5, seeds 810 and 811. "Authors' pipeline" is their repository at
    81f8995 run end to end; "Port" is this class with a re-implementation of
    their probe.

    ===============  ========  ======  =================  ======
    Task             Metric    Paper   Authors' pipeline  Port
    ===============  ========  ======  =================  ======
    DS42             macro-F1  0.7018  0.7000             0.7039
    DS42             AUROC     0.7840  0.7840             0.7840
    DS19             macro-F1  0.5598  0.5599             0.5304
    DS19             AUROC     0.5683  0.5687             0.5549
    ===============  ========  ======  =================  ======

    Both pipelines build the same windows (1628 and 1190), on which this class
    gives the authors' encoder features (max-abs difference 0.0, Base and
    Large, eval mode). On the authors' features, with their row order and their
    head-initialisation order, the port's probe gives DS19 macro-F1 0.5599. In
    training mode the dropout draws differ from the reference.

    **Pretrained weights.** ``geonirs_base.pt`` and ``geonirs_large.pt`` are in
    the ``checkpoint/`` folder (Git LFS) of `GeoNIRS/GeoNIRS@81f8995
    <https://github.com/GeoNIRS/GeoNIRS/tree/81f8995e5c3aa6b70e55812b194fc04e4e7ea5b7/checkpoint>`_.
    :meth:`load_state_dict` takes the file as released and drops the decoders.
    They expect 5 Hz signals preprocessed as above and load for ``n_times`` up
    to 150; for Large, pass the arguments of its column in the table.

    .. code-block:: python

        import torch
        from braindecode.models import GeoNIRS

        payload = torch.load("geonirs_base.pt", map_location="cpu", weights_only=True)
        model = GeoNIRS(chs_info=raw.info["chs"], n_outputs=2, n_times=60, sfreq=5.0)
        model.load_state_dict(payload)

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
        patch_size: int = 10,
        max_patches: int = 15,
        on_non_divisible: Literal["error", "pad", "crop"] = "error",
        embed_dim: int = 192,
        num_heads: int = 8,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        geometry_hidden_dim: int | None = None,
        geometry_heads: int | None = None,
        n_latent_channels: int = 16,
        compressor_heads: int | None = None,
        feature_dim: int = 64,
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
        self.patch_size = patch_size
        self.n_patches = patch_count(self.n_times, patch_size, on_non_divisible)
        self.min_valid_samples = math.ceil(
            patch_size * float(minimum_patch_valid_fraction)
        )
        self.n_latent_channels = n_latent_channels
        self.pool = pool

        temporal_encoder = dict(
            patch_size=patch_size,
            max_patches=max(max_patches, self.n_patches),
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_prob=drop_prob,
            activation=activation,
        )
        self.hbo_encoder = _TemporalEncoder(**temporal_encoder)
        self.hbr_encoder = _TemporalEncoder(**temporal_encoder)
        self.geometry_encoder = None
        if use_channel_geometry:
            self.geometry_encoder = _GeometryEncoder(
                embed_dim=embed_dim,
                hidden_dim=geometry_hidden_dim or embed_dim,
                num_heads=geometry_heads or num_heads,
                drop_prob=drop_prob,
                activation=activation,
            )
        self.channel_compressor = _ChannelCompressorAttention(
            embed_dim=embed_dim,
            n_latent_channels=n_latent_channels,
            num_heads=compressor_heads or num_heads,
            drop_prob=drop_prob,
            activation=activation,
        )
        self.hbo_projection = _chromophore_projection(
            embed_dim, feature_dim, drop_prob, activation
        )
        self.hbr_projection = _chromophore_projection(
            embed_dim, feature_dim, drop_prob, activation
        )
        self.register_buffer(
            "default_geometry",
            torch.from_numpy(_optode_geometry(self._chs_info, self.channel_order)),
            persistent=False,
        )
        self.split_patches = Rearrange(
            "batch pairs (patches samples) -> batch pairs patches samples",
            samples=patch_size,
        )
        self.pair_over_time = Rearrange("batch pairs -> batch pairs 1")
        self.patch_over_latents = Rearrange("batch patches -> batch 1 patches")
        self.patch_over_features = Rearrange("batch patches -> batch 1 patches 1")
        self.flatten_foundation = Rearrange(
            "batch chrom latents patches dim -> batch (chrom latents patches dim)"
        )
        self.join_chromophores = Rearrange(
            "batch chrom latents patches dim -> batch latents patches (chrom dim)"
        )
        self.valid_over_features = Rearrange(
            "batch latents patches -> batch latents patches 1"
        )
        self.mean_over_patches = Rearrange("batch latents dim -> batch latents 1 dim")
        self.flatten_statistics = Rearrange(
            "stat batch latents dim -> batch (latents stat dim)"
        )
        n_features = {
            "flatten": 2 * n_latent_channels * self.n_patches * feature_dim,
            "mean_std": 4 * n_latent_channels * feature_dim,
        }[pool]
        self.head_dropout = nn.Dropout(head_drop_prob)
        self.final_layer = nn.Linear(n_features, self.n_outputs)

    def encode(
        self,
        x: torch.Tensor,
        geometry: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """``(batch, n_chans, n_times)`` -> ``"foundation"``
        ``(batch, 2, M, P, feature_dim)`` (:math:`Z^O, Z^R`) and its
        validity ``"valid"`` ``(batch, M, P)``. ``geometry`` is
        ``(n_pairs, 6)`` or ``(batch, n_pairs, 6)``; ``ch_mask``
        ``(batch, n_pairs)`` flags real pairs.
        """
        # (batch, n_chans, n_times) -> (batch, n_chans, P * S)
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
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
        latent_hbo, latent_hbr = self.channel_compressor(
            temporal_hbo, temporal_hbr, geometry_embedding, patch_valid
        )
        # (batch, n_pairs, P) -> (batch, P): a patch with a valid pair
        time_valid = patch_valid.any(dim=1)
        # (batch, P) -> (batch, 1, P, 1)
        valid_scale = self.patch_over_features(time_valid)
        valid_scale = valid_scale.to(latent_hbo.dtype)
        # (batch, M, P, D) -> (batch, M, P, F): Z^O, Z^R
        compact_hbo = self.hbo_projection(latent_hbo)
        compact_hbo = compact_hbo * valid_scale
        compact_hbr = self.hbr_projection(latent_hbr)
        compact_hbr = compact_hbr * valid_scale
        # 2 x (batch, M, P, F) -> (batch, 2, M, P, F)
        foundation = torch.stack([compact_hbo, compact_hbr], dim=1)
        # (batch, P) -> (batch, 1, P) -> (batch, M, P)
        valid = self.patch_over_latents(time_valid)
        valid = valid.expand(-1, self.n_latent_channels, -1)
        return {"foundation": foundation, "valid": valid}

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        n_features = self.final_layer.in_features
        self.final_layer = nn.Linear(n_features, self.n_outputs)

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
        patch_size,
        max_patches,
        embed_dim,
        num_heads,
        depth,
        mlp_ratio,
        drop_prob,
        activation,
    ):
        super().__init__()
        self.split_patches = Rearrange(
            "batch pairs (patches samples) -> batch pairs patches samples",
            samples=patch_size,
        )
        self.patch_projection = nn.Linear(patch_size, embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.time_embedding = nn.Embedding(max_patches, embed_dim)
        mlp_dim = round(embed_dim * mlp_ratio)
        self.temporal_transformer = _transformer(
            embed_dim, num_heads, mlp_dim, drop_prob, activation, depth
        )
        self.output_norm = nn.LayerNorm(embed_dim)
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

    def __init__(self, embed_dim, hidden_dim, num_heads, drop_prob, activation):
        super().__init__()
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.spatial_transformer = _transformer(
            embed_dim, num_heads, embed_dim * 2, drop_prob, activation, 1
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


class _ChannelCompressorAttention(nn.Module):
    """The geometry-aware channel compressor (the authors'
    ``BrainOmniChannelCompressor``): cross-attention of
    ``n_latent_channels`` learned queries over the channels, then an output
    projection and a feed-forward block.

    Private because one softmax over the fused HbO/HbR keys weighs both
    chromophores' values (one dropout mask), and the released checkpoints
    name its projections ``q_proj``/``k_proj``/``v_proj``/``out_proj``.
    """

    def __init__(self, embed_dim, n_latent_channels, num_heads, drop_prob, activation):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        latent_queries = torch.empty(n_latent_channels, embed_dim)
        self.latent_queries = nn.Parameter(latent_queries)
        norms = (nn.LayerNorm(embed_dim) for _ in range(3))
        self.q_norm, self.k_norm, self.v_norm = norms
        projections = (nn.Linear(embed_dim, embed_dim) for _ in range(4))
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = projections
        self.attn_dropout = nn.Dropout(drop_prob)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            activation(),
            nn.Dropout(drop_prob),
            nn.Linear(embed_dim * 2, embed_dim),
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
        self.latents_first = Rearrange(
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
        latent_hbo = self._output_block(
            compressed[0], latents, active_weight, batch_size, n_patches
        )
        latent_hbr = self._output_block(
            compressed[1], latents, active_weight, batch_size, n_patches
        )
        return latent_hbo, latent_hbr

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
        hidden = self.latents_first(hidden)
        return hidden.contiguous()


class _ResidualProjectionBlock(nn.Module):
    """The authors' ``ResidualProjectionBlock``: ``skip(x) + residual_scale *
    main(x)``, ``residual_scale`` starting at 0.1.

    Private because no shared block has a learned residual scale on a
    width-changing skip projection.
    """

    def __init__(self, input_dim, output_dim, drop_prob, activation):
        super().__init__()
        self.skip = nn.Linear(input_dim, output_dim)
        self.main = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            activation(),
            nn.Dropout(drop_prob),
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


def _transformer(embed_dim, num_heads, mlp_dim, drop_prob, activation, depth):
    """Pre-norm ``nn.TransformerEncoder`` with a final LayerNorm."""
    layer = nn.TransformerEncoderLayer(
        embed_dim,
        num_heads,
        mlp_dim,
        drop_prob,
        activation(),
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, depth, norm=nn.LayerNorm(embed_dim), enable_nested_tensor=False
    )


def _chromophore_projection(embed_dim, feature_dim, drop_prob, activation):
    """Chromophore-specific residual projection from :math:`D` to :math:`F`:
    three residual stages, ``[192, 128, 96, 64]`` for GeoNIRS-Base and
    ``[D, D / 2, D / 4, F]`` otherwise (``[384, 192, 96, 128]`` for Large),
    as the paper's Table 6."""
    widths = [embed_dim, embed_dim // 2, embed_dim // 4, feature_dim]
    if embed_dim == 192 and feature_dim == 64:
        widths = [192, 128, 96, 64]
    stages = [
        _ResidualProjectionBlock(widths[i], widths[i + 1], drop_prob, activation)
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
