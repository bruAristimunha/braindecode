# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3-Clause
# Adapted from https://github.com/GeoNIRS/GeoNIRS model/jung_inspired_mae.py (81f8995;
# no licence file; paper arXiv:2610.17845), the GeoNIRS authors' paper-guided
# implementation of Jung et al. (2025).

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.functional import (
    pad_or_crop_to_patches,
    patch_count,
    sinusoidal_positional_encoding,
)
from braindecode.functional.patches import _check_on_non_divisible
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import fnirs_channel_order, fnirs_split_hbo_hbr


class JungInspiredMAE(EEGModuleMixin, nn.Module):
    r"""The GeoNIRS authors' Jung-inspired masked autoencoder (Li et al., 2026).

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    Jung et al. :footcite:`jung2025mae` "propose a novel foundation model for
    fNIRS data based on a self-supervised masked autoencoder framework", and
    report that "a larger masking size aligns with the low-frequency nature of
    fNIRS signals". They released no code or weights. The GeoNIRS authors
    :footcite:`li2026geonirs` "evaluate a paper-guided implementation of the
    multi-dataset fNIRS pretraining framework of Jung et al. (2025), denoted
    as Jung-inspired MAE. To accommodate heterogeneous channel counts,
    Jung-inspired MAE uses channel-independent encoding with shared
    parameters but does not model physical channel geometry"; it is "a
    paper-guided implementation rather than an exact reproduction of the
    original method". This class ports that implementation (their
    ``jung_inspired_vanilla_mae``) and loads the weights they trained,
    ``jung_mae.pt``; it is not code by Jung et al.

    Each HbO/HbR pair is encoded on its own: its samples are cut into patches
    of ``patch_samples`` HbO and HbR samples, each patch is projected to a
    token, sinusoidal patch positions are added, and a pre-norm Transformer
    encodes the tokens of the pair. A linear head reads all the tokens, as the
    benchmark's ``flatten_linear`` head (whose training-set standardization of
    the features is left to the caller). NaN marks a missing sample: patches
    with too few finite samples are left out of attention and zeroed.

    :meth:`load_state_dict` takes the GeoNIRS ``jung_mae.pt`` (decoder
    dropped). The file holds no data configuration; the GeoNIRS benchmark
    feeds it the same rows as :class:`GeoNIRS`: 5 Hz HbO/HbR in 150-sample
    (30 s) windows, each recording's HbO and HbR z-scored separately.

    Parameters
    ----------
    patch_samples : int, default=10
        Samples per patch and chromophore (2 s at 5 Hz).
    on_non_divisible : {"error", "pad", "crop"}, default="error"
        A ``n_times`` that is not a multiple of ``patch_samples`` is refused,
        as by the reference, zero-padded or cropped.
    encoder_dim : int, default=192
        Token width.
    encoder_layers : int, default=4
        Transformer layers.
    encoder_heads : int, default=8
        Attention heads per layer.
    mlp_ratio : float, default=4.0
        Feed-forward width relative to ``encoder_dim``.
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_C, HbR_1..HbR_C]`` or MNE's ``[HbO_1, HbR_1, ...]``;
        ``None`` reads it from ``chs_info``, and is ``"stacked"`` without it.
        ``n_chans`` must be even.
    drop_prob : float, default=0.1
        Dropout in the encoder layers.
    head_drop_prob : float, default=0.1
        Dropout before ``final_layer``.
    minimum_patch_valid_fraction : float, default=0.8
        Fraction of finite samples a patch needs to be valid.
    activation : type[nn.Module], default=nn.GELU
        Activation of the encoder feed-forward sub-layers.

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
        on_non_divisible: Literal["error", "pad", "crop"] = "error",
        encoder_dim: int = 192,
        encoder_layers: int = 4,
        encoder_heads: int = 8,
        mlp_ratio: float = 4.0,
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
        self.channel_order = fnirs_channel_order(self._chs_info, channel_order)
        if self.n_chans % 2:
            raise ValueError(
                "JungInspiredMAE needs HbO/HbR pairs: n_chans must be even."
            )
        _check_on_non_divisible(on_non_divisible)
        self.on_non_divisible = on_non_divisible
        self.patch_samples = patch_samples
        self.min_valid_samples = math.ceil(patch_samples * minimum_patch_valid_fraction)
        n_patches = patch_count(self.n_times, patch_samples, on_non_divisible)
        self.patch_embed = nn.Linear(2 * patch_samples, encoder_dim)
        layer = nn.TransformerEncoderLayer(
            encoder_dim,
            encoder_heads,
            int(encoder_dim * mlp_ratio),
            drop_prob,
            activation(),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, encoder_layers, nn.LayerNorm(encoder_dim), enable_nested_tensor=False
        )
        self.sample_over_chromophores = Rearrange(
            "batch pairs time -> batch pairs time 1"
        )
        self.split_patches = Rearrange(
            "batch pairs (patches samples) chrom -> (batch pairs) patches (samples chrom)",
            samples=patch_samples,
        )
        self.split_patch_mask = Rearrange(
            "batch pairs (patches samples) -> batch pairs patches samples",
            samples=patch_samples,
        )
        self.merge_pairs = Rearrange("batch pairs patches -> (batch pairs) patches")
        self.patch_over_features = Rearrange("rows patches -> rows patches 1")
        self.flatten_tokens = Rearrange(
            "batch pairs patches dim -> batch (pairs patches dim)"
        )
        self.head_dropout = nn.Dropout(head_drop_prob)
        self.final_layer = nn.Linear(
            self.n_chans // 2 * n_patches * encoder_dim, self.n_outputs
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        # Xavier-uniform weights, zero biases, as the GeoNIRS reference.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)

    def encode(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """``(batch, n_chans, n_times)`` -> ``"foundation"`` ``(batch,
        n_chans // 2, n_patches, encoder_dim)`` (zero at invalid patches) and
        ``"valid"`` ``(batch, n_chans // 2, n_patches)``. ``mask`` (bool, same
        shape as ``x``): True samples count as missing."""
        if mask is not None:
            x = x.masked_fill(mask, float("nan"))
        # (batch, n_chans, n_times) -> (batch, n_chans, n_patches * patch_samples)
        x = pad_or_crop_to_patches(x, self.patch_samples, self.on_non_divisible)
        # -> 2 x (batch, n_pairs, n_patches * patch_samples)
        hbo, hbr = fnirs_split_hbo_hbr(x, self.channel_order)
        batch_size, n_pairs, _ = hbo.shape
        hbo_finite = torch.isfinite(hbo)
        hbr_finite = torch.isfinite(hbr)
        # (batch, n_pairs, n_patches * patch_samples): both chromophores finite
        valid = hbo_finite & hbr_finite
        # -> (batch, n_pairs, n_patches * patch_samples, 2)
        signal = torch.stack((hbo, hbr), dim=-1)
        valid_over_chromophores = self.sample_over_chromophores(valid)
        signal = torch.where(valid_over_chromophores, signal, 0.0)
        # -> (batch * n_pairs, n_patches, 2 * patch_samples): [HbO_t, HbR_t, ...]
        patches = self.split_patches(signal)
        # (batch, n_pairs, n_patches * patch_samples) -> (batch, n_pairs, n_patches, patch_samples)
        valid_patches = self.split_patch_mask(valid)
        # -> (batch, n_pairs, n_patches)
        n_valid_samples = valid_patches.sum(dim=-1)
        patch_valid = n_valid_samples >= self.min_valid_samples
        # -> (batch * n_pairs, n_patches)
        rows_valid = self.merge_pairs(patch_valid)
        n_patches = patches.shape[1]
        # (batch * n_pairs, n_patches, 2 * patch_samples) -> (batch * n_pairs, n_patches, encoder_dim)
        tokens = self.patch_embed(patches)
        # (n_patches, encoder_dim)
        positions = sinusoidal_positional_encoding(
            n_patches, tokens.shape[-1], device=x.device, dtype=tokens.dtype
        )
        tokens = tokens + positions
        # A pair without valid patch attends to all (an all-masked softmax
        # is NaN); its output is zeroed.
        row_active = rows_valid.any(dim=-1, keepdim=True)
        attend = rows_valid | ~row_active
        padding = ~attend
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        # (batch * n_pairs, n_patches) -> (batch * n_pairs, n_patches, 1)
        weight = self.patch_over_features(rows_valid)
        weight = weight.to(encoded.dtype)
        encoded = encoded * weight
        # (batch * n_pairs, n_patches, encoder_dim) -> (batch, n_pairs, n_patches, encoder_dim)
        foundation = encoded.unflatten(0, [batch_size, n_pairs])
        return {"foundation": foundation, "valid": patch_valid}

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)
        nn.init.xavier_uniform_(self.final_layer.weight)  # as the constructor
        nn.init.zeros_(self.final_layer.bias)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Also takes the released payload (weights under ``"model"``): the
        decoder is dropped, and without ``final_layer`` the head is kept."""
        state_dict = state_dict.get("model", state_dict)
        decoder = ("decoder_embed.", "mask_token", "decoder.", "prediction.")
        cleaned = {k: v for k, v in state_dict.items() if not k.startswith(decoder)}
        if not any(k.startswith("final_layer.") for k in cleaned):
            head = self.final_layer.state_dict(prefix="final_layer.")
            cleaned.update(head)
        return super().load_state_dict(cleaned, strict=strict, assign=assign)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """``x``: ``(batch, n_chans, n_times)``, NaN = missing; ``mask`` as in
        :meth:`encode`. ``ch_mask`` is refused: ``final_layer`` spans every
        channel (mark a missing channel with NaN)."""
        if ch_mask is not None:
            raise ValueError(
                "JungInspiredMAE does not accept ch_mask; use NaN channels."
            )
        # (batch, n_chans, n_times) -> (batch, n_pairs, n_patches, encoder_dim)
        foundation = self.encode(x, mask)["foundation"]
        # -> (batch, n_pairs * n_patches * encoder_dim)
        features = self.flatten_tokens(foundation)
        features = self.head_dropout(features)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)
