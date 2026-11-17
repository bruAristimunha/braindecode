# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
#
# Adapted from https://github.com/microsoft/MtReNIRS (MIT).

from __future__ import annotations

import math
from collections import OrderedDict
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from braindecode.functional import (
    apply_rotary,
    check_ch_mask,
    pad_or_crop_to_patches,
    patch_count,
    rotary_angles,
    rotary_frequencies,
    scaled_dot_product_attention,
)
from braindecode.functional.patches import _check_on_non_divisible
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import (
    extract_channel_locations_from_chs_info,
    fnirs_chromophores_from_chs_info,
)
from braindecode.modules import FeedForwardBlock, TimmAttention


class MtReNIRS(EEGModuleMixin, nn.Module, license="mit"):
    r"""MtReNIRS from Chae et al. (2026) :footcite:`microsoft2026mtrenirs`.

    :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    .. figure:: https://braindecode.org/dev/_static/model/mtrenirs.png
       :align: center
       :alt: MtReNIRS preprocessing, montage harmonization, token embedding and pretraining

       Fig. 2 of the paper (`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_):
       "(A) fNIRS pre-processing pipeline, (B) harmonization pipeline of
       heterogeneous montages, (C) deriving signal and spatio-chromophore token
       embedding, and (D) masked pretraining and downstream transfer". This
       class is the encoder of (C)-(D) with a linear head.

    A montage-aware masked autoencoder for fNIRS. Each token is one 10-s window
    :math:`x_{c,t} \in \mathbb{R}^{100}` of a chromophore channel
    :math:`c = (i, b_c)`; "Each 10-s signal segment was projected by a linear
    layer to a 128-dimensional signal embedding", to which an MLP of six
    Fourier bands of :math:`u_c = [q_c^\top, b_c]^\top` (10-05 slot position
    and HbO/HbR flag) is added. Six pre-norm Transformer blocks with per-head
    QK LayerNorm follow, and "rotary position encoding (RoPE) with base 500
    encoded the temporal-window index" :footcite:`su2024roformer`.

    **Mapping.** Input ``(batch, n_chans, n_times)``, one row per HbO or HbR
    channel in any order, cut into ``n_times // window_samples`` windows.
    :math:`q_c` is ``loc[:3]`` (metres) of ``chs_info``, read as given (the
    reference first snaps each channel to its nearest 10-05 slot,
    ``mtrenirs-alignment``); :math:`b_c` comes from ``coil_type`` or the
    channel name. Without locations, pass ``coordinates`` to :meth:`forward`.
    Channels without a location and windows with a NaN are left out of the
    attention, as in the reference. The default ``pool="temporal_mean"``
    feeds ``final_layer`` the paper's linear-probe input (one window per
    trial); ``"channel_temporal_mean"`` is added here for ``ch_mask``. The
    decoder and masking of pretraining are left out. State-dict keys are the
    authors' (no ``mapping``), and :meth:`load_state_dict` takes the released
    ``{"config", "model"}`` payload strictly; the head keeps its init.

    =============  ==================  ======================  ===================
    Token length   ``window_samples``  SujinBak LOSO macro-F1  Weights
    =============  ==================  ======================  ===================
    2 s            20                  62.3                    not released
    5 s            50                  65.0                    not released
    10 s           100                 67.9                    ``mtrenirs_10s.pt``
    =============  ==================  ======================  ===================

    .. important::
       ``mtrenirs_10s.pt`` (MIT) is mirrored at `braindecode/mtrenirs-pretrained
       <https://huggingface.co/braindecode/mtrenirs-pretrained>`_. It expects
       10 Hz HbO/HbR, preprocessed as in Notes, in 100-sample windows:

       .. code-block:: python

           import torch
           from huggingface_hub import hf_hub_download

           from braindecode.models import MtReNIRS

           path = hf_hub_download(
               "braindecode/mtrenirs-pretrained",
               "mtrenirs_10s.pt",
               revision="bbc68ce5e00a4515081d900227fc06d111858e75",
           )
           payload = torch.load(path, map_location="cpu", weights_only=True)
           # raw: MNE Raw with the 10-05 slot positions in loc[:3]
           model = MtReNIRS(chs_info=raw.info["chs"], n_outputs=3, n_times=100, sfreq=10.0)
           model.load_state_dict(payload)

    `License <https://github.com/microsoft/MtReNIRS/blob/main/LICENSE.txt>`_

    Parameters
    ----------
    window_samples : int, default=100
        Samples per token window :math:`x_{c,t}` (10 s at 10 Hz); ``data.t_win``.
    d_model : int, default=128
        Token and Transformer width; ``model.d_model``.
    depth : int, default=6
        Transformer blocks; ``model.encoder.depth``.
    n_heads : int, default=4
        Attention heads, with an even ``d_model // n_heads`` for RoPE;
        ``model.encoder.n_heads``.
    mlp_hidden : int, default=512
        Feed-forward width of each block; ``model.encoder.mlp_hidden``.
    fourier_num_bands : int, default=6
        Fourier bands :math:`2^k \pi`, :math:`k < 6`, per coordinate of
        :math:`u_c` (48 features); ``model.fourier_num_bands``.
    spatial_mlp_hidden : int, default=128
        Hidden width of the 48 -> 128 -> 128 spatio-chromophore MLP;
        ``model.spatial_mlp_hidden``.
    rope_base : float, default=500.0
        RoPE base of the window index; ``model.rope_base``.
    qk_norm : bool, default=True
        Per-head LayerNorm of queries and keys; ``model.qk_norm``.
    pool : {"temporal_mean", "channel_temporal_mean", "flatten"}, default="temporal_mean"
        Readout of the ``(batch, n_windows, n_chans, d_model)`` tokens: mean
        over valid windows, over valid windows and channels (takes
        ``ch_mask``), or all tokens (the reference's ``"flatten"``).
    on_non_divisible : {"crop", "pad", "error"}, default="crop"
        A short last window is dropped (the reference's ``remainder="drop"``),
        zero-padded (``"pad"``) or refused.
    pos_center : tuple of float
        Centre (m) subtracted from :math:`q_c`, computed on the pretraining
        corpus; the checkpoint's ``data.pos_center``.
    pos_scale : float
        RMS scale (m) dividing the centred :math:`q_c`; ``data.pos_scale``.
    drop_prob : float, default=0.1
        Dropout of attention weights, attention output and MLP;
        ``model.encoder.dropout``.
    activation : type[nn.Module], default=nn.GELU
        Activation of the spatio-chromophore MLP and feed-forward blocks.

    Notes
    -----
    Protocol (paper Sec. III): MBLL (PPF 6.0), "band-pass filtered each run
    from 0.01 to 0.5 Hz, resampled to 10 Hz", per-channel z-score over the run,
    clip to :math:`[-6, 6]`, windows [onset, onset + 10 s). Pretraining: 12
    datasets (268 subjects, 567 h), 60-s crops, 25 % of tokens masked, AdamW,
    100 epochs, batch 128, lr :math:`2 \times 10^{-4}`; epoch 95 selected on
    the validation loss. Evaluation on SujinBak 2019 :footcite:`bak2019ufft`
    (30 subjects, 3 classes): LOSO with ~15 % inner-validation subjects, 50
    epochs, AdamW lr :math:`10^{-4}`, batch 32, best validation macro-F1; LP,
    MLP (64 units), LoRA (r 8, :math:`\alpha` 16), Full-FT, Scratch-FT; and
    2-8-shot linear probes (200 full-batch updates, 50 draws per subject).

    The band-pass must remove content above 0.5 Hz: the released
    ``from_mne_raw`` uses MNE's default FIR (0.87 of the amplitude passes at
    1 Hz), which gives 61.1 (LOSO LP) and 35.1 (2-shot) instead of 68.3 and
    40.3. A zero-phase Butterworth reproduces the paper (:gh:`1564`):
    ``raw.filter(0.01, 0.5, method="iir", iir_params=dict(order=3, ftype="butter"))``.

    Replication with this port and ``mtrenirs_10s.pt`` (macro-F1, %; LP/MLP
    3 seeds, others 1; channels on their nearest 10-05 slots around C3/C4;
    the authors' encoder gives identical features):

    =========  =====  ====  ====  =======  ==========  ======  ======  ======  ======
    \          LP     MLP   LoRA  Full-FT  Scratch-FT  2-shot  4-shot  6-shot  8-shot
    =========  =====  ====  ====  =======  ==========  ======  ======  ======  ======
    Paper      67.9   67.2  70.3  70.2     66.7        40.4    46.2    50.0    53.0
    Port       68.3   67.4  69.3  69.6     67.8        40.3    46.6    50.4    53.0
    =========  =====  ====  ====  =======  ==========  ======  ======  ======  ======

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
        window_samples: int = 100,
        d_model: int = 128,
        depth: int = 6,
        n_heads: int = 4,
        mlp_hidden: int = 512,
        fourier_num_bands: int = 6,
        spatial_mlp_hidden: int = 128,
        rope_base: float = 500.0,
        qk_norm: bool = True,
        pool: Literal[
            "temporal_mean", "channel_temporal_mean", "flatten"
        ] = "temporal_mean",
        on_non_divisible: Literal["crop", "pad", "error"] = "crop",
        pos_center: tuple[float, float, float] = (
            0.00021245056996122003,
            -0.022675372660160065,
            0.020025914534926414,
        ),
        pos_scale: float = 0.09265176206827164,
        drop_prob: float = 0.1,
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
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) "
                "with an even head_dim for rotary position embeddings."
            )
        pools = ("temporal_mean", "channel_temporal_mean", "flatten")
        if pool not in pools:
            raise ValueError(f"pool must be one of {pools}, got {pool!r}.")
        _check_on_non_divisible(on_non_divisible)
        self.window_samples = window_samples
        self.pool = pool
        self.on_non_divisible = on_non_divisible
        self.n_windows = patch_count(self.n_times, window_samples, on_non_divisible)
        self.embed = _TokenEmbedding(
            window_samples, d_model, fourier_num_bands, spatial_mlp_hidden, activation
        )
        dropout = float(drop_prob)
        blocks = [
            _TransformerBlock(
                d_model, n_heads, mlp_hidden, dropout, rope_base, qk_norm, activation
            )
            for _ in range(depth)
        ]
        self.encoder = _Encoder(blocks, d_model)
        self.register_buffer(
            "default_coordinates",
            torch.from_numpy(_coordinates(self._chs_info, pos_center, pos_scale)),
            persistent=False,
        )
        self.split_windows = Rearrange(
            "batch chans (windows samples) -> batch windows chans samples",
            samples=window_samples,
        )
        self.chan_over_coordinates = Rearrange("batch chans -> batch chans 1")
        self.chan_over_windows = Rearrange("batch chans -> batch 1 chans")
        self.merge_tokens = Rearrange(
            "batch windows chans samples -> batch (windows chans) samples"
        )
        self.merge_token_mask = Rearrange(
            "batch windows chans -> batch (windows chans)"
        )
        self.coordinates_over_windows = Rearrange(
            "batch chans coords -> batch 1 chans coords"
        )
        self.merge_coordinates = Rearrange(
            "batch windows chans coords -> batch (windows chans) coords"
        )
        self.token_over_features = Rearrange("batch tokens -> batch tokens 1")
        self.valid_over_features = Rearrange(
            "batch windows chans -> batch windows chans 1"
        )
        self.flatten_channels = Rearrange("batch chans dim -> batch (chans dim)")
        self.flatten_tokens = Rearrange(
            "batch windows chans dim -> batch (windows chans dim)"
        )
        self.feature_dim = {
            "temporal_mean": self.n_chans * d_model,
            "channel_temporal_mean": d_model,
            "flatten": self.n_windows * self.n_chans * d_model,
        }[pool]
        self.final_layer = nn.Linear(self.feature_dim, self.n_outputs)

    def encode_windows(
        self,
        x: torch.Tensor,
        coordinates: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(batch, n_chans, n_times)`` -> tokens ``(batch, n_windows, n_chans,
        d_model)`` and their validity ``(batch, n_windows, n_chans)``.
        ``coordinates`` is ``(n_chans, 4)`` or ``(batch, n_chans, 4)``
        (:math:`u_c`, positions already standardized); NaN positions mark
        channels without a location."""
        batch_size, n_chans, _ = x.shape
        if coordinates is None:
            coord = self.default_coordinates
            if coord.numel() == 0:
                raise ValueError(
                    "MtReNIRS needs channel positions and chromophores: chs_info "
                    "with channel locations (loc[:3]) or the coordinates argument."
                )
            if coord.shape[0] != n_chans:
                raise ValueError(
                    "x and chs_info have different channel counts: pass the "
                    "coordinates of the channels of x."
                )
        else:
            coord = coordinates
            if coord.shape[-2] != n_chans:
                raise ValueError(
                    "coordinates and x have different channel counts: pass one "
                    "row per channel of x."
                )
        # (batch, n_chans, n_times) -> (batch, n_chans, n_windows * window_samples)
        x = pad_or_crop_to_patches(x, self.window_samples, self.on_non_divisible)
        # -> (batch, n_windows, n_chans, window_samples): x_{c,t}
        windows = self.split_windows(x)
        n_windows = windows.shape[1]
        finite = torch.isfinite(windows)
        # (batch, n_windows, n_chans): a NaN masks its window
        valid = finite.all(dim=-1)
        coord = coord.to(device=x.device, dtype=windows.dtype)
        # (n_chans, 4) or (batch, n_chans, 4) -> (batch, n_chans, 4)
        coord = coord.expand(batch_size, -1, -1)
        positions = coord[..., :3]
        positions_finite = torch.isfinite(positions)
        # (batch, n_chans): a channel without a location is masked, at position 0
        located = positions_finite.all(dim=-1)
        located_over_coordinates = self.chan_over_coordinates(located)
        coord_zeros = torch.zeros_like(coord)
        coord = torch.where(located_over_coordinates, coord, coord_zeros)
        # (batch, n_chans) -> (batch, 1, n_chans)
        located_over_windows = self.chan_over_windows(located)
        valid = valid & located_over_windows
        if ch_mask is not None:
            ch_mask = self.chan_over_windows(ch_mask)
            ch_mask = ch_mask.to(torch.bool)
            valid = valid & ch_mask
        # (batch, n_windows, n_chans) -> (batch, n_windows * n_chans)
        valid_tokens = self.merge_token_mask(valid)
        window_zeros = torch.zeros_like(windows)
        tokens = torch.where(finite, windows, window_zeros)
        # (batch, n_windows, n_chans, window_samples) -> (batch, n_windows * n_chans, window_samples)
        tokens = self.merge_tokens(tokens)
        # (batch, n_chans, 4) -> (batch, n_windows * n_chans, 4): u_c of every token
        coord_tokens = self.coordinates_over_windows(coord)
        coord_tokens = coord_tokens.expand(-1, n_windows, -1, -1)
        coord_tokens = self.merge_coordinates(coord_tokens)
        # (batch, n_windows * n_chans): the RoPE position of a token is its window t
        window_indices = torch.arange(n_windows, device=x.device)
        window_indices = window_indices.repeat_interleave(n_chans)
        window_indices = window_indices.expand(batch_size, -1)
        # -> (batch, n_windows * n_chans, d_model): z_{c,t}
        embedded = self.embed(tokens, coord_tokens)
        encoded = self.encoder(embedded, window_indices, valid_tokens)
        token_weight = self.token_over_features(valid_tokens)
        token_weight = token_weight.to(encoded.dtype)
        encoded = encoded * token_weight
        # (batch, n_windows * n_chans, d_model) -> (batch, n_windows, n_chans, d_model)
        encoded = encoded.unflatten(1, [n_windows, n_chans])
        return encoded, valid

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.feature_dim, n_outputs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Also takes the official checkpoint (``{"model": ...}``); a missing
        head keeps its init."""
        state_dict = dict(state_dict.get("model", state_dict))
        if not any(k.startswith("final_layer.") for k in state_dict):
            state_dict.update(self.final_layer.state_dict(prefix="final_layer."))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        coordinates: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """``(batch, n_chans, n_times)`` -> ``(batch, n_outputs)``.

        ``coordinates`` as in :meth:`encode_windows`. ``ch_mask``
        ``(batch, n_chans)`` (:func:`~braindecode.functional.check_ch_mask`)
        needs ``pool="channel_temporal_mean"``; padded channels are left out of
        the attention and the pooling.
        """
        ch_mask = check_ch_mask(ch_mask, x)
        if ch_mask is not None and self.pool != "channel_temporal_mean":
            raise ValueError(
                "ch_mask needs pool='channel_temporal_mean': the other heads "
                "are tied to the channels."
            )
        # (batch, n_chans, n_times) -> (batch, n_windows, n_chans, d_model)
        encoded, valid = self.encode_windows(x, coordinates, ch_mask)
        if self.pool == "flatten":
            # -> (batch, n_windows * n_chans * d_model)
            features = self.flatten_tokens(encoded)
        else:
            # mean over the valid windows (and channels)
            dims: List[int] = [1] if self.pool == "temporal_mean" else [1, 2]
            # (batch, n_windows, n_chans) -> (batch, n_windows, n_chans, 1)
            weight = self.valid_over_features(valid)
            weight = weight.to(encoded.dtype)
            count = weight.sum(dim=dims)
            count = count.clamp_min(1.0)
            weighted = encoded * weight
            total = weighted.sum(dim=dims)
            # -> (batch, n_chans, d_model) or (batch, d_model)
            features = total / count
            if self.pool == "temporal_mean":
                # (batch, n_chans, d_model) -> (batch, n_chans * d_model)
                features = self.flatten_channels(features)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)


class _TokenEmbedding(nn.Module):
    """Signal embedding of the window plus the spatio-chromophore embedding
    (an MLP of the Fourier features of :math:`u_c`)."""

    def __init__(self, window_samples, d_model, num_bands, hidden, activation):
        super().__init__()
        # ``signal.proj`` and ``spatial.mlp`` are the checkpoint's key paths
        self.signal = nn.Sequential(
            OrderedDict(proj=nn.Linear(window_samples, d_model))
        )
        # 2^k pi, k = 0, ..., num_bands - 1
        freqs = (2.0 ** torch.arange(num_bands).float()) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)
        mlp = nn.Sequential(
            nn.Linear(4 * num_bands * 2, hidden),
            activation(),
            nn.Linear(hidden, d_model),
        )
        self.spatial = nn.Sequential(OrderedDict(mlp=mlp))
        self.coordinates_over_bands = Rearrange(
            "batch tokens coords -> batch tokens coords 1"
        )
        self.merge_bands = Rearrange(
            "batch tokens coords bands -> batch tokens (coords bands)"
        )

    def forward(self, signal: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(device=coordinates.device, dtype=coordinates.dtype)
        # (batch, n_tokens, 4) -> (batch, n_tokens, 4, 1)
        coordinates = self.coordinates_over_bands(coordinates)
        # -> (batch, n_tokens, 4, num_bands): 2^k pi u_{c,j}
        angles = coordinates * freqs
        # -> (batch, n_tokens, 4 * num_bands)
        angles = self.merge_bands(angles)
        # (batch, n_tokens, window_samples) -> (batch, n_tokens, d_model)
        signal_embedding = self.signal(signal)
        sine = angles.sin()
        cosine = angles.cos()
        # -> (batch, n_tokens, 2 * 4 * num_bands): phi(u_c)
        fourier = torch.cat([sine, cosine], dim=-1)
        # -> (batch, n_tokens, d_model)
        spatial_embedding = self.spatial(fourier)
        return signal_embedding + spatial_embedding


class _RotaryAttention(TimmAttention):
    """The authors' ``MultiHeadAttention``: timm attention with per-head QK
    LayerNorm, a rotary embedding of the window index and a key padding mask.

    Private because :class:`~braindecode.modules.TimmAttention`'s forward takes
    neither per-token positions nor a mask; its ``qkv``/``proj`` are reused.
    """

    def __init__(self, dim, heads, dropout, rope_base, qk_norm):
        super().__init__(dim, heads, qkv_bias=True, attn_drop=dropout)
        self.rope_base = float(rope_base)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.angles_over_heads = Rearrange("batch tokens dim -> batch 1 tokens dim")
        self.keys_over_queries = Rearrange("batch tokens -> batch 1 1 tokens")
        self.merge_heads = Rearrange(
            "batch heads tokens dim -> batch tokens (heads dim)"
        )

    # takes positions and valid, unlike timm's forward: mypy's override check is off
    def forward(  # type: ignore[override]
        self, values: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        # (batch, n_tokens, dim) -> 3 x (batch, heads, n_tokens, head_dim)
        query, key, value = self._heads(values)
        query = self.q_norm(query)
        key = self.k_norm(key)
        # (head_dim // 2,)
        inv_freq = rotary_frequencies(
            self.head_dim, self.rope_base, positions.device, query.dtype
        )
        positions = positions.to(query.dtype)
        # (batch, n_tokens) -> (batch, n_tokens, head_dim)
        angles = rotary_angles(positions, inv_freq, interleaved=False)
        cosine = angles.cos()
        sine = angles.sin()
        # (batch, n_tokens, head_dim) -> (batch, 1, n_tokens, head_dim)
        cosine = self.angles_over_heads(cosine)
        sine = self.angles_over_heads(sine)
        query = apply_rotary(query, cosine, sine, interleaved=False)
        key = apply_rotary(key, cosine, sine, interleaved=False)
        # (batch, n_tokens): a sample with no valid token attends to all of them
        any_valid = valid.any(dim=-1, keepdim=True)
        attend = valid | ~any_valid
        # (batch, n_tokens) -> (batch, 1, 1, n_tokens)
        attend = self.keys_over_queries(attend)
        bias = torch.zeros_like(attend, dtype=query.dtype)
        blocked = ~attend
        bias = bias.masked_fill(blocked, float("-inf"))
        dropout_p = self.attn_drop.p if self.training else 0.0
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=dropout_p,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        # (batch, heads, n_tokens, head_dim) -> (batch, n_tokens, dim)
        output = self.merge_heads(output)
        return self.proj(output)


class _TransformerBlock(nn.Module):
    """Pre-norm block: QK-norm rotary self-attention, then feed-forward.

    Private because its attention takes per-token positions and a validity mask
    (:class:`~braindecode.modules.PreNormBlock` calls ``attn(x)`` only) and the
    released ``mtrenirs_10s.pt`` keys follow this layout.
    """

    def __init__(
        self, d_model, n_heads, mlp_hidden, dropout, rope_base, qk_norm, activation
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _RotaryAttention(d_model, n_heads, dropout, rope_base, qk_norm)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FeedForwardBlock(
            d_model, 1, dropout, activation, hidden_features=mlp_hidden
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, values: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        # (batch, n_tokens, d_model) -> (batch, n_tokens, d_model)
        normed = self.norm1(values)
        attended = self.attn(normed, positions, valid)
        attended = self.drop(attended)
        values = values + attended
        normed = self.norm2(values)
        update = self.mlp(normed)
        return values + update


class _Encoder(nn.Module):
    """The Transformer blocks and a final LayerNorm."""

    def __init__(self, blocks: list[nn.Module], d_model: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, values: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        # (batch, n_tokens, d_model) -> (batch, n_tokens, d_model)
        for block in self.blocks:
            values = block(values, positions, valid)
        return self.norm(values)


def _coordinates(chs_info, pos_center, pos_scale) -> np.ndarray:
    """``(n_chans, 4)`` :math:`u_c`: ``(loc[:3] - pos_center) / pos_scale``
    (NaN when unknown or all zero) and 0 for HbO, 1 for HbR, as the official
    ``chromophore_map``; ``(0, 4)`` without ``chs_info`` or any location."""
    if chs_info is None:
        return np.zeros((0, 4), np.float32)
    chromophores = fnirs_chromophores_from_chs_info(chs_info)
    if "unknown" in chromophores:
        name = chs_info[chromophores.index("unknown")].get("ch_name")
        raise ValueError(f"MtReNIRS needs HbO/HbR channels, got {name!r}.")
    loc = extract_channel_locations_from_chs_info(chs_info, fill_missing=True)
    loc = np.asarray(loc)
    loc[(loc == 0).all(axis=1)] = np.nan  # MNE without a montage: NaN or zeros
    if not np.isfinite(loc).any():  # forward raises unless coordinates are given
        return np.zeros((0, 4), np.float32)
    center = np.asarray(pos_center, dtype=np.float32)
    xyz = (loc - center) / float(pos_scale)
    is_hbr = np.equal(chromophores, "hbr")
    return np.c_[xyz, is_hbr].astype(np.float32)
