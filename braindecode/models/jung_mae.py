# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# Adapted from https://github.com/GeoNIRS/GeoNIRS model/jung_inspired_mae.py (81f8995),
# the GeoNIRS authors' implementation of Jung, Lee and An (2025).

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


class JungMAE(EEGModuleMixin, nn.Module):
    r"""JungMAE, the GeoNIRS authors' masked autoencoder after Jung et al. (2025).

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    The architecture follows Jung, Lee and An (IEEE BCI 2025,
    doi:10.1109/BCI65088.2025.10931275), who "propose a novel foundation model
    for fNIRS data based on a self-supervised masked autoencoder framework"
    :footcite:`jung2025mae` and released no code. This class ports the GeoNIRS
    authors' re-implementation (``model/`` of GeoNIRS at ``81f8995``), "a
    paper-guided implementation rather than an exact reproduction of the
    original method", and loads ``jung_mae.pt``, which the GeoNIRS authors
    trained; their paper calls it Jung-MAE (Tables 14 and 15)
    :footcite:`li2026geonirs`. It encodes channels independently with shared
    parameters and does not model channel geometry.

    Each HbO/HbR pair is cut into patches of ``patch_samples`` time points;
    the HbO and HbR samples of a patch are interleaved
    (``[HbO_t, HbR_t, ...]``), projected to ``encoder_dim`` (``patch_embed``)
    and given a sinusoidal patch position. A pre-norm Transformer
    (``encoder``) encodes the patches of each pair, with the same weights for
    every pair. A patch with fewer than ``minimum_patch_valid_fraction`` valid
    samples is left out of the attention and its token is zero. Pretraining
    (not ported) hid half of the valid patches of each pair and reconstructed
    them with a 2-layer, 96-wide decoder under a mean squared error. The head
    is the benchmark's ``flatten_linear``: all tokens flattened, then Dropout
    (``head_dropout``) and one Linear (``final_layer``).

    **Braindecode mapping.** ``x`` is ``(batch, n_chans, n_times)``,
    ``n_chans = 2 * n_pairs``: HbO rows then HbR rows, or MNE's interleaved
    order read from ``chs_info``. The reference takes ``hbo``/``hbr`` tensors
    with validity masks; here NaN marks a missing sample (an all-NaN row, a
    missing channel) and the opt-in ``mask`` marks more. :meth:`encode`
    returns the reference's ``"foundation"``
    ``(batch, n_pairs, n_patches, encoder_dim)`` and ``"valid"``. With
    ``jung_mae.pt`` the features equal the reference's in eval mode (max-abs
    0.0, with and without missing samples); in training mode the reference
    packs the visible tokens and the port masks keys, so dropout draws differ.
    The benchmark's training-set standardization of the features is left to
    the caller. ``final_layer`` is drawn as the reference's other Linear
    layers (Xavier uniform, zero bias); the benchmark head keeps PyTorch's
    default. ``on_non_divisible="pad"``/``"crop"`` and ``mask`` are additions
    of the port; ``ch_mask`` is refused, as ``final_layer`` spans every pair.

    .. important::
       **Pre-trained weights.** ``checkpoint/jung_mae.pt`` of
       github.com/GeoNIRS/GeoNIRS at ``81f8995`` (sha256 ``6bb13e85...``), not
       on the Hugging Face Hub. :meth:`load_state_dict` takes the payload as
       released, drops the decoder and keeps the new head:

       .. code-block:: python

           payload = torch.load("jung_mae.pt", map_location="cpu", weights_only=True)
           model = JungMAE(n_chans=40, n_outputs=2, n_times=60, sfreq=5.0)
           model.load_state_dict(payload)

       It stores the model configuration (the defaults here), no data
       configuration. The GeoNIRS paper feeds every model the same windows:
       HbO and HbR (µmol/L) resampled to 5 Hz and each z-scored per recording
       over all valid channels and samples; 30 s windows for pretraining,
       each task's length (10 s or 12 s) downstream.

    Parameters
    ----------
    patch_samples : int, default=10
        Reference ``patch_samples``: samples per patch (2 s at 5 Hz).
    on_non_divisible : {"error", "pad", "crop"}, default="error"
        A ``n_times`` that is not a multiple of ``patch_samples`` is refused
        (as the reference), zero-padded or cropped.
    encoder_dim : int, default=192
        Reference ``encoder_dim``: token width.
    encoder_layers : int, default=4
        Reference ``encoder_layers``: Transformer layers.
    encoder_heads : int, default=8
        Reference ``encoder_heads``: attention heads per layer.
    mlp_ratio : float, default=4.0
        Reference ``mlp_ratio``: feed-forward width over ``encoder_dim``.
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_P, HbR_1..HbR_P]`` or MNE's ``[HbO_1, HbR_1, ...]``;
        ``None`` reads it from ``chs_info`` (``"stacked"`` without it).
        ``n_chans`` must be even.
    drop_prob : float, default=0.1
        Reference ``dropout`` in the encoder layers.
    head_drop_prob : float, default=0.1
        Benchmark ``head_dropout``, before ``final_layer``.
    minimum_patch_valid_fraction : float, default=0.8
        Reference name: fraction of valid samples a patch needs.
    activation : type[nn.Module], default=nn.GELU
        Feed-forward activation (reference: ``"gelu"``).

    Notes
    -----
    Protocol of the GeoNIRS benchmark (paper Appendix B.5-B.6): eight
    held-out binary tasks, subject-level five-fold cross-validation, seeds
    810 and 811; 30 epochs, batch 16, AdamW (weight decay 0.05), 3 warmup
    epochs and cosine decay, class-weighted cross-entropy, early stopping
    (patience 15), model selection by validation balanced accuracy, one test
    evaluation. The linear probe (LP) trains the head only (learning rate
    1e-4); LoRA (rank 4, alpha 8) adapts every Linear.

    Replication of Jung-MAE-LP with ``jung_mae.pt`` in this port and the
    benchmark's linear-probe loop (seeds 810, 811, mean over folds and
    seeds); the reference encoder gives the same features (max-abs 0.0):

    =====================  ========  =================  ========
    Task                   Metric    GeoNIRS paper (%)  Port (%)
    =====================  ========  =================  ========
    DS42 (action mode)     macro-F1  68.36              68.35
    DS42 (action mode)     AUROC     76.78              76.80
    DS19 (pain intensity)  macro-F1  54.23              54.95
    DS19 (pain intensity)  AUROC     58.39              58.63
    =====================  ========  =================  ========

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
                "JungMAE needs HbO/HbR pairs: n_chans must be even."
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
                "JungMAE does not accept ch_mask; use NaN channels."
            )
        # (batch, n_chans, n_times) -> (batch, n_pairs, n_patches, encoder_dim)
        foundation = self.encode(x, mask)["foundation"]
        # -> (batch, n_pairs * n_patches * encoder_dim)
        features = self.flatten_tokens(foundation)
        features = self.head_dropout(features)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)
