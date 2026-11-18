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
    r"""JungMAE from Jung et al. (2025), as implemented by the GeoNIRS authors.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    Labelled fNIRS data are slow and costly to collect, and supervised
    decoders need a lot of them. Jung, Lee and An :footcite:`jung2025mae`
    "propose a novel foundation model for fNIRS data based on a
    self-supervised masked autoencoder framework", pretrained on unlabelled
    recordings. They report that it matches supervised training "while
    requiring only one-third of the labeled training data", and that "a larger
    masking size aligns with the low-frequency nature of fNIRS signals". Their
    paper (IEEE BCI 2025) is not open access and came without code, so what is
    said about it here comes from its abstract.

    This class ports the implementation and the ``jung_mae.pt`` weights of the
    GeoNIRS authors (``GeoNIRS/GeoNIRS`` at commit ``81f8995``), who use the
    model as a baseline and call it Jung-MAE (Tables 1, 14 and 15 of
    :footcite:`li2026geonirs`). Their paper describes it as "a paper-guided
    implementation of the multi-dataset fNIRS pretraining framework of Jung et
    al. (2025)" which, to "accommodate heterogeneous channel counts", "uses
    channel-independent encoding with shared parameters" (Section 3.1). The
    details below are read from the GeoNIRS code.

    .. rubric:: Architectural Overview

    Every HbO/HbR pair (one source-detector channel) is a sequence of its own.
    Its window is cut into 2 s patches, each patch becomes one token that
    holds both chromophores, and a small Transformer encodes the tokens of
    that pair. The same weights serve every pair, which is how one checkpoint
    handles montages of any size. The classifier then flattens the tokens of
    all the pairs. With :math:`B` the batch size, :math:`C` = ``n_chans // 2``
    pairs, :math:`Q` = ``patch_size``, :math:`P` patches and :math:`D` =
    ``embed_dim``, the model has the following sequence of steps::

        x (B, 2C, n_times): HbO and HbR rows, NaN = missing sample
        1. HbO/HbR split, invalid samples to zero  -> (B, C, P*Q) each
        2. to_patches: [HbO_t, HbR_t, ...]         -> (B*C, P, 2Q)
        3. patch_embed + sinusoidal position       -> (B*C, P, D)
        4. encoder, invalid patches not attended   -> (B, C, P, D)   encode()
        5. flatten_layer, head_dropout             -> (B, C*P*D)
        6. final_layer                             -> (B, n_outputs)

    .. rubric:: Macro Components

    - ``JungMAE.patch_embed`` **Patch tokenization**

      A patch interleaves the :math:`Q` HbO samples :math:`o` and HbR samples
      :math:`r` of pair :math:`c`. One :class:`~torch.nn.Linear` maps it to a
      :math:`D`-wide token, and the fixed sinusoidal encoding of the patch
      index :math:`p` is added:

      .. math::

          \mathbf{z}_{c,p} = \mathbf{W}\,[o_{c,pQ},\, r_{c,pQ},\, \dots,\,
          o_{c,pQ+Q-1},\, r_{c,pQ+Q-1}] + \mathbf{b} + \mathrm{PE}(p).

    - **Patch validity** (``minimum_patch_valid_fraction``)

      A sample is valid when both chromophores are finite and it is not in
      ``mask``. Invalid samples are set to zero, and a patch counts only when
      at least :math:`\lceil 0.8\,Q \rceil` of its samples are valid. The
      GeoNIRS paper applies the same 80 % rule to its own model (Appendix
      B.1.2).

    - ``JungMAE.encoder`` **Transformer encoder**

      Four pre-norm :class:`~torch.nn.TransformerEncoderLayer` blocks (8
      heads, feed-forward width 768, GELU) and a final
      :class:`~torch.nn.LayerNorm`, over the :math:`P` tokens of one pair.
      Invalid patches are not attended and their output is zero, so a pair
      without any valid patch gives zero features.

    - ``JungMAE.final_layer`` **Classification head**

      Flatten, :class:`~torch.nn.Dropout` and one :class:`~torch.nn.Linear`,
      the ``flatten_linear`` head of the GeoNIRS benchmark. It is not
      pretrained, and its input size fixes ``n_chans`` and ``n_times``.

    .. rubric:: Temporal, Spatial, and Chromophore Encoding

    - **Temporal.** Non-overlapping 2 s patches (10 samples at 5 Hz) with a
      fixed sinusoidal position. The GeoNIRS paper pretrains on 30 s windows
      (15 patches) and keeps each downstream task's 10 s or 12 s window
      (Appendix B.1.2).
    - **Spatial.** None. Pairs never attend to each other and the model sees
      no optode position. A pair is identified only by where its features sit
      in the head input, so the pairs must keep the order used in training.
    - **Chromophores.** HbO and HbR share each token, sample by sample; HbT
      is not used. The rows must be the HbO and the HbR rows of the same
      pairs (see ``channel_order``).

    .. rubric:: Pretraining

    In the reference, each pair keeps half of its valid patches visible,
    drawn at random, and hides the others. The encoder reads the visible
    tokens only. A lighter decoder (two layers of width 96, with a learned
    mask token) predicts every patch, and the loss is the mean squared error
    over the valid samples of the hidden patches. The decoder is stored in
    ``jung_mae.pt`` and is not part of this class.

    .. rubric:: Known Limitations

    - The model "does not model physical channel geometry" (GeoNIRS paper,
      Section 3.1): different channel counts "are supported without modeling
      physical channel coordinates or cross-layout correspondence" (Table 13).
    - It is "a paper-guided implementation rather than an exact reproduction
      of the original method" (GeoNIRS paper, Appendix G).

    .. rubric:: Pretrained Weights

    ``checkpoint/jung_mae.pt`` of ``GeoNIRS/GeoNIRS`` at ``81f8995`` was
    trained by the GeoNIRS authors and is not on the Hugging Face Hub. It
    holds the encoder (1,783,872 parameters), the decoder (244,436) and the
    model configuration, which equals the defaults of this class.

    .. important::
       :meth:`load_state_dict` takes the file as released. It drops the
       decoder and keeps the new ``final_layer``:

       .. code-block:: python

           import torch
           from braindecode.models import JungMAE

           payload = torch.load("jung_mae.pt", map_location="cpu", weights_only=True)
           model = JungMAE(n_chans=40, n_outputs=2, n_times=60, sfreq=5.0)
           model.load_state_dict(payload)

    The weights expect the inputs of the GeoNIRS paper (Appendix B.1): HbO and
    HbR in µmol/L resampled to 5 Hz, each chromophore z-scored per recording
    over all its valid channels and time points.

    .. rubric:: Differences from the Reference Implementation

    - **Input.** One tensor ``(batch, n_chans, n_times)``, with the HbO rows
      then the HbR rows, or MNE's interleaved order read from ``chs_info``.
      The reference takes ``hbo`` and ``hbr`` tensors with validity masks.
      Here NaN marks a missing sample (an all-NaN row is a missing channel),
      and the opt-in ``mask`` marks more.
    - **Parity.** The reference packs the valid tokens of each pair; this
      class masks the invalid ones as attention keys. With ``jung_mae.pt``
      the features equal those of the authors' module (max-abs 0.0) in eval
      mode under :func:`torch.no_grad`. With gradients on they differ by
      about 2e-6 at most (float32 rounding), and in training mode the
      dropout draws differ.
    - **Head.** The benchmark head standardizes the flattened features with
      training-set statistics before its Dropout; this head does not, so
      standardize the eval-mode ``return_features`` yourself to reproduce
      the benchmark. ``final_layer`` starts Xavier-uniform with a zero bias,
      as the other Linear layers of the reference.
    - **Additions.** ``on_non_divisible="pad"`` or ``"crop"`` (the reference
      refuses a partial patch) and ``mask``. ``ch_mask`` is refused, because
      ``final_layer`` spans every pair.

    Parameters
    ----------
    patch_size : int, default=10
        Samples per patch (``patch_samples`` in the reference), 2 s at 5 Hz.
    embed_dim : int, default=192
        Width of the tokens (``encoder_dim`` in the reference).
    depth : int, default=4
        Number of Transformer layers (``encoder_layers`` in the reference).
    num_heads : int, default=8
        Attention heads per layer (``encoder_heads`` in the reference).
    mlp_ratio : float, default=4.0
        Feed-forward width over ``embed_dim``.
    drop_prob : float, default=0.1
        Dropout of the encoder layers (``dropout`` in the reference).
    head_drop_prob : float, default=0.1
        Dropout before ``final_layer`` (``head_dropout`` in the benchmark).
    minimum_patch_valid_fraction : float, default=0.8
        Fraction of valid samples a patch needs, in (0, 1].
    activation : type[nn.Module], default=nn.GELU
        Activation of the feed-forward layers.
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_C, HbR_1..HbR_C]`` or MNE's ``[HbO_1, HbR_1, ...]``.
        ``None`` reads it from ``chs_info`` (``"stacked"`` without it).
        ``n_chans`` must be even.
    on_non_divisible : {"error", "pad", "crop"}, default="error"
        What to do when ``n_times`` is not a multiple of ``patch_size``:
        raise, as the reference does; pad with missing samples, to which the
        validity rule applies; or crop.

    Notes
    -----
    The GeoNIRS benchmark (Section 3.1 and Appendices B.5-B.6 of
    :footcite:`li2026geonirs`) evaluates eight held-out binary tasks with
    subject-level five-fold cross-validation and the seeds 810 and 811. Every
    method gets the same windows and folds and the same training: 30 epochs,
    AdamW, class-weighted cross-entropy and model selection by validation
    balanced accuracy. Jung-MAE-LP trains the head only, with a learning rate
    of 1e-4.

    This class with ``jung_mae.pt`` and the benchmark's linear-probe loop
    reproduces Jung-MAE-LP (mean over the five test folds and the two seeds):

    =====================  ========  =================  ==============
    Task                   Metric    GeoNIRS paper (%)  This class (%)
    =====================  ========  =================  ==============
    DS42 (action mode)     macro-F1  68.36              68.35
    DS42 (action mode)     AUROC     76.78              76.80
    DS19 (pain intensity)  macro-F1  54.23              54.95
    DS19 (pain intensity)  AUROC     58.39              58.63
    =====================  ========  =================  ==============

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
        on_non_divisible: Literal["error", "pad", "crop"] = "error",
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 8,
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
            raise ValueError("JungMAE needs HbO/HbR pairs: n_chans must be even.")
        _check_on_non_divisible(on_non_divisible)
        self.on_non_divisible = on_non_divisible
        self.patch_size = patch_size
        if not 0 < minimum_patch_valid_fraction <= 1:
            raise ValueError("minimum_patch_valid_fraction must be in (0, 1].")
        self.min_valid_samples = math.ceil(patch_size * minimum_patch_valid_fraction)
        n_patches = patch_count(self.n_times, patch_size, on_non_divisible)
        self.patch_embed = nn.Linear(2 * patch_size, embed_dim)
        layer = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            int(embed_dim * mlp_ratio),
            drop_prob,
            activation(),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, depth, nn.LayerNorm(embed_dim), enable_nested_tensor=False
        )
        self.sample_over_chromophores = Rearrange(
            "batch pairs time -> batch pairs time 1"
        )
        self.split_patches = Rearrange(
            "batch pairs (patches samples) chrom"
            " -> (batch pairs) patches (samples chrom)",
            samples=patch_size,
        )
        self.split_patch_mask = Rearrange(
            "batch pairs (patches samples) -> batch pairs patches samples",
            samples=patch_size,
        )
        self.merge_pairs = Rearrange("batch pairs patches -> (batch pairs) patches")
        self.patch_over_features = Rearrange("rows patches -> rows patches 1")
        self.flatten_tokens = Rearrange(
            "batch pairs patches dim -> batch (pairs patches dim)"
        )
        self.head_dropout = nn.Dropout(head_drop_prob)
        self.final_layer = nn.Linear(
            self.n_chans // 2 * n_patches * embed_dim, self.n_outputs
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
        n_chans // 2, n_patches, embed_dim)`` (zero at invalid patches) and
        ``"valid"`` ``(batch, n_chans // 2, n_patches)``. ``mask`` (bool, same
        shape as ``x``): True samples count as missing."""
        if mask is not None:
            x = x.masked_fill(mask, float("nan"))
        n_samples = x.shape[-1]
        # (batch, n_chans, n_times) -> (batch, n_chans, n_patches * patch_size)
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        # padded samples are missing: the validity threshold applies to them
        padded = torch.arange(x.shape[-1], device=x.device) >= n_samples
        x = x.masked_fill(padded, float("nan"))
        # -> 2 x (batch, n_pairs, n_patches * patch_size)
        hbo, hbr = fnirs_split_hbo_hbr(x, self.channel_order)
        batch_size, n_pairs, _ = hbo.shape
        hbo_finite = torch.isfinite(hbo)
        hbr_finite = torch.isfinite(hbr)
        # (batch, n_pairs, n_patches * patch_size): both chromophores finite
        valid = hbo_finite & hbr_finite
        # -> (batch, n_pairs, n_patches * patch_size, 2)
        signal = torch.stack((hbo, hbr), dim=-1)
        valid_over_chromophores = self.sample_over_chromophores(valid)
        signal = torch.where(valid_over_chromophores, signal, 0.0)
        # -> (batch * n_pairs, n_patches, 2 * patch_size): [HbO_t, HbR_t, ...]
        patches = self.split_patches(signal)
        # (batch, n_pairs, n_patches * patch_size)
        # -> (batch, n_pairs, n_patches, patch_size)
        valid_patches = self.split_patch_mask(valid)
        # -> (batch, n_pairs, n_patches)
        n_valid_samples = valid_patches.sum(dim=-1)
        patch_valid = n_valid_samples >= self.min_valid_samples
        # -> (batch * n_pairs, n_patches)
        rows_valid = self.merge_pairs(patch_valid)
        n_patches = patches.shape[1]
        # (batch * n_pairs, n_patches, 2 * patch_size)
        # -> (batch * n_pairs, n_patches, embed_dim)
        tokens = self.patch_embed(patches)
        # (n_patches, embed_dim)
        # at least float32, then cast: as the reference in half precision
        table_dtype = torch.promote_types(tokens.dtype, torch.float32)
        positions = sinusoidal_positional_encoding(
            n_patches, tokens.shape[-1], device=x.device, dtype=table_dtype
        )
        positions = positions.to(tokens.dtype)
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
        # (batch * n_pairs, n_patches, embed_dim)
        # -> (batch, n_pairs, n_patches, embed_dim)
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
            raise ValueError("JungMAE does not accept ch_mask; use NaN channels.")
        # (batch, n_chans, n_times) -> (batch, n_pairs, n_patches, embed_dim)
        foundation = self.encode(x, mask)["foundation"]
        # -> (batch, n_pairs * n_patches * embed_dim)
        features = self.flatten_tokens(foundation)
        features = self.head_dropout(features)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)
