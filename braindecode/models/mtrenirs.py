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
    r"""MtReNIRS from Chae et al. (2026).

    :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    .. figure:: https://braindecode.org/dev/_static/model/mtrenirs.png
       :align: center
       :alt: MtReNIRS preprocessing, montage harmonization, token embedding and pretraining

       Fig. 2 of Chae et al. (2026), arXiv:2610.36159 (`CC BY 4.0
       <https://creativecommons.org/licenses/by/4.0/>`_): "Overview of (A)
       fNIRS pre-processing pipeline, (B) harmonization pipeline of
       heterogeneous montages, (C) deriving signal and spatio-chromophore token
       embedding, and (D) masked pretraining and downstream transfer operations
       of the foundation model". This class is the encoder of (C)-(D) with a
       linear head.

    MtReNIRS (Montage-aligned transferable Representations for fNIRS) is the
    fNIRS foundation model of Chae et al. :footcite:`microsoft2026mtrenirs`.
    From the paper (Abstract): "a montage-aware masked autoencoder, pretrained
    on 12 public fNIRS datasets spanning 268 subjects and 567 recording hours,
    learns generalizable neurovascular representations that transfer across
    three external datasets. [...] Our fNIRS foundation model, MtReNIRS,
    jointly encodes the slow hemodynamic nature of fNIRS, chromophore identity
    and cortical location by mapping heterogeneous montages onto a shared
    anatomical reference." The two
    constraints behind the design (Sec. I): "Spatial channels across different
    montages must be aligned to a common anatomical reference so that each
    fNIRS signal can be consistently related to the underlying cortical
    activity", and "HbO/HbR signals are slow, delayed, and highly
    autocorrelated". A token is therefore one 10-s window of one chromophore
    channel, tagged with the canonical 10-05 position :footcite:`oostenveld2001five`
    of its source-detector (S-D) pair and with its chromophore; six Transformer
    blocks mix all tokens, with the window index as their only time code.

    .. rubric:: Architecture Overview

    With :math:`B` the batch, :math:`C` the HbO/HbR rows (``n_chans``),
    :math:`P` = ``window_samples``, :math:`W` = ``n_times // window_samples``
    windows, :math:`N = W C` tokens and :math:`D` = ``embed_dim``, the model
    has the following sequence of steps::

        input x                                                (B, C, n_times)
        - Windowing (split_windows): 10-s windows x_{c,t};
          a short last window is cropped (on_non_divisible)    (B, W, C, P)
        - Token validity: a window with a NaN, a channel
          without a location or a ch_mask-padded channel       (B, W, C)
        - Token sequence (merge_tokens), window-major          (B, N, P)
        - Signal embedding (embed.signal.proj): Linear P -> D  (B, N, D)
        - + Spatio-chromophore embedding (embed.spatial.mlp):
          Fourier features of u_c, MLP 48 -> 128 -> D          (B, N, D)
        - Transformer encoder (encoder.blocks), depth x:
          LayerNorm, QK-norm RoPE self-attention, residual;
          LayerNorm, MLP D -> mlp_dim -> D, residual           (B, N, D)
        - Final LayerNorm (encoder.norm); invalid tokens zeroed (B, N, D)
        - Token grid (output of encode_windows)                (B, W, C, D)
        - Readout (pool): "temporal_mean"                      (B, C * D)
                          "channel_temporal_mean"              (B, D)
                          "flatten"                            (B, W * C * D)
        - Head (final_layer): Linear                           (B, n_outputs)

    With the released 10-s weights and one 10-s trial (``n_times=100`` at
    10 Hz), :math:`W = 1`, :math:`P = 100`, :math:`D = 128`; the 40 HbO/HbR
    rows of SujinBak 2019 give 40 x 128 = 5120 features for the linear probe.

    .. rubric:: Macro Components

    - **Tokens** (``split_windows``, ``merge_tokens``). The paper (Sec.
      III-C): "We represented each token by a chromophore channel :math:`c`
      and temporal window :math:`t`, where :math:`c = (i, b_c)` identifies an
      S-D pair :math:`i` and chromophore type :math:`b_c \in \{0, 1\}`, with 0
      and 1 denoting HbO and HbR, respectively." Each token holds the
      :math:`P = 100` samples :math:`x_{c,t} \in \mathbb{R}^{100}` of one
      window; there is no CLS token.
    - **Signal embedding** (``embed.signal.proj``, ``nn.Linear(100, 128)``).
      "Each 10-s signal segment :math:`x_{c,t} \in \mathbb{R}^{100}` was
      projected by a linear layer to a 128-dimensional signal embedding":

      .. math::

         e^{\mathrm{sig}}_{c,t} = W_x\, x_{c,t} + b_x,
         \qquad W_x \in \mathbb{R}^{128 \times 100}.

    - **Spatio-chromophore embedding** (``embed.spatial.mlp``,
      ``nn.Linear(48, 128)``, GELU, ``nn.Linear(128, 128)``). "For channel
      :math:`c = (i, b_c)`, the canonical spatial coordinate is inherited from
      its S-D pair, :math:`q_c = q_{s_i} \in \mathbb{R}^3`. We concatenated
      this coordinate with the binary chromophore indicator :math:`b_c` to
      form :math:`u_c = [q_c^\top, b_c]^\top \in \mathbb{R}^4`. We encoded the
      resulting :math:`u_c` with six bands of Fourier features"
      :footcite:`tancik2020fourier` (Eq. 2), passed through "A 48 -> 128 ->
      128 multilayer perceptron (MLP) with GELU activation" and summed with
      the signal embedding into the token :math:`z_{c,t} \in \mathbb{R}^{128}`:

      .. math::

         \phi(u_c) = \left[\sin(2^k \pi u_{c,j}),\, \cos(2^k \pi u_{c,j})
         \right]_{j = 1, \dots, 4}^{k = 0, \dots, 5} \in \mathbb{R}^{48},

      .. math::

         z_{c,t} = e^{\mathrm{sig}}_{c,t}
         + W_2\, \mathrm{GELU}(W_1 \phi(u_c) + b_1) + b_2.

      "This embedding lets the encoder distinguish region-specific activity
      while modeling coupled, often opposing HbO/HbR dynamics." As in the
      authors' ``FourierFeatures``, the 48 features are the 24 sines followed
      by the 24 cosines, the six bands of each coordinate together.
    - **Canonical coordinates** (``default_coordinates``, from ``chs_info``;
      or the ``coordinates`` argument of :meth:`forward`). Before the model,
      the paper maps every montage to the 10-05 system (Sec. III-B, Fig. 2B):
      "For valid 3-D geometries provided by the dataset, we centered the raw
      S-D midpoint coordinates :math:`p_i` and matched their root mean square
      (RMS) radius to the standard 10-05 reference (3-D RMS alignment; Fig.
      3B)" (Eq. 1), then "assigned each aligned midpoint to its nearest canonical slot":

      .. math::

         \tilde{p}_i = (p_i - \bar{p})\, \frac{s_q}{s_p} + \bar{q},
         \qquad s_p^2 = \frac{1}{N} \sum_i \lVert p_i - \bar{p} \rVert_2^2,

      with :math:`N` the number of valid S-D midpoints, :math:`\bar{p}`,
      :math:`\bar{q}` the centres of the raw and canonical coordinates and
      :math:`s_p`, :math:`s_q` their RMS distances. Datasets with only 2-D
      layout schematics get a "2-D anchor placement" on "an anatomically
      supported local tangent plane", and those "with known placements but
      missing or degenerate geometry" a label assignment. "Our model used the fixed coordinate
      :math:`q_{s_i}` of assigned slot :math:`s_i`, not the intermediate
      aligned coordinate. We normalized all available canonical coordinates
      with one center and RMS scale computed from the pretraining corpus."
      This class applies that last step,
      :math:`\hat{q}_c = (q_c - \mu) / \sigma` with :math:`\mu` =
      ``coordinate_center`` and :math:`\sigma` = ``coordinate_scale``; Eq. (1)
      and the slot assignment run before it (the authors' MIT
      ``mtrenirs.alignment.align``, slots of MNE's ``standard_1005``
      montage).
    - **Transformer encoder** (``encoder.blocks``, ``encoder.norm``). "The
      encoder contained six pre-layer-normalized Transformer blocks with a
      128-dimensional state, four attention heads, a 512-dimensional MLP, and
      0.1 dropout. Per-head layer normalization stabilized queries and keys.
      To preserve cross-window order without conflating time with spatial
      identity, rotary position encoding (RoPE) :footcite:`su2024roformer`
      with base 500 encoded the temporal-window index." A block
      (``_TransformerBlock``) computes

      .. math::

         h = z + \mathrm{Drop}\big(\mathrm{Attn}(\mathrm{LN}_1(z))\big),
         \qquad z' = h + \mathrm{MLP}(\mathrm{LN}_2(h)),

      with ``mlp`` = Linear(128, 512), GELU, Dropout, Linear(512, 128).
      In each of the :math:`H = 4` heads of size :math:`d_h = 32`
      (``attn.qkv``, ``attn.q_norm``, ``attn.k_norm``, ``attn.proj``), the
      query and key of the normalized input :math:`y_n` of token :math:`n`,
      in window :math:`t_n`, are layer-normalized, then rotated by their
      window index:

      .. math::

         \tilde{q}_n = \mathcal{R}_{t_n} \mathrm{LN}_q(W_q y_n), \quad
         \tilde{k}_m = \mathcal{R}_{t_m} \mathrm{LN}_k(W_k y_m), \quad
         \mathrm{Attn}_n = \sum_m \mathrm{softmax}_m\!\left(
         \frac{\tilde{q}_n^\top \tilde{k}_m}{\sqrt{d_h}} + M_m \right) W_v y_m,

      where :math:`M_m = 0` for a valid key and :math:`-\infty` otherwise.
      The rotation, from the authors' ``_rope_angles`` and ``_apply_rope``
      (rotate-half layout), with :math:`\beta` = ``rope_base``:

      .. math::

         \mathcal{R}_t v = v \odot \cos(t \theta) + \mathrm{rot}(v) \odot
         \sin(t \theta), \quad \theta = [\theta_0, \dots,
         \theta_{d_h/2-1}, \theta_0, \dots, \theta_{d_h/2-1}], \quad
         \theta_j = \beta^{-2j/d_h},

      with :math:`\mathrm{rot}(v) = [-v_{d_h/2:}, v_{:d_h/2}]`. "All S-D pairs
      from the same window shared that index; spatial identity entered only
      through the canonical coordinate embedding." A final LayerNorm closes
      the encoder; "The inference encoder contained 1.23 million parameters"
      (1,226,368 in the authors' model card).
    - **Readout and head** (``pool``, ``final_layer``). The paper evaluates a
      linear probe on frozen features (Sec. III-D): "we temporally averaged
      encoder outputs before classification so that the feature
      dimensionality remained fixed. For the primary 10-s model, each trial
      contained a single window, making temporal-mean and flatten pooling
      equivalent." ``"temporal_mean"`` (the default) is that readout, one
      ``embed_dim`` vector per channel, concatenated; ``final_layer`` is the
      linear head.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - *Temporal:* within a window, time is the 100 input weights of the
      signal projection (no convolution); across windows, RoPE of the window
      index :math:`t` only. The weights expect 10 Hz and 10-s windows; the
      paper chose 10 s "to capture the multi-second rise and peak of the local
      hemodynamic response while preserving finer temporal structure" (Sec.
      III-A).
    - *Spatial:* no channel embedding table. Every row of ``x`` is one
      chromophore channel, in any order and number; the HbO and HbR rows of
      one S-D pair share :math:`q_c`, read from ``loc[:3]`` of ``chs_info``
      (metres, head frame, already on a 10-05 slot). Rows without a location
      (NaN or all-zero ``loc``) are left out of the attention and the
      pooling. Without positions in ``chs_info``, pass ``coordinates``
      (:math:`u_c` with :math:`\hat{q}_c`, shape ``(n_chans, 4)`` or
      ``(batch, n_chans, 4)``) to :meth:`forward`.
    - *Chromophore:* :math:`b_c = 0` for HbO and 1 for HbR, as the
      checkpoint's ``chromophore_map``, read from MNE's ``coil_type`` or the
      ``" hbo"``/``" hbr"`` name suffix; any other channel (e.g. HbT) raises.
    - *Spectral:* none explicit; the inputs are band-limited to 0.01-0.5 Hz by
      the preprocessing (Notes).

    .. rubric:: Additional Mechanisms

    - **Masked pretraining** (Fig. 2D, Sec. III-C; not part of this class).
      "For each 60-s crop, we randomly masked 25% of valid tokens and passed
      only visible tokens to the encoder. A two-block lightweight Transformer
      decoder reconstructed each masked 10-s waveform from the encoded visible
      tokens and learned mask tokens. We minimized mean squared error only
      over masked real tokens" (Eq. 3), with :math:`\mathcal{M}` the masked
      tokens:

      .. math::

         \mathcal{L}_{\mathrm{MAE}} = \frac{1}{100\, |\mathcal{M}|}
         \sum_{n \in \mathcal{M}} \lVert \hat{x}_n - x_n \rVert_2^2.

    - **Token masking.** Invalid tokens are excluded as keys and zeroed at
      the output, as the authors' ``key_padding_mask``; ``ch_mask`` adds
      padded channels to them (``pool="channel_temporal_mean"``).
    - **Adaptation strategies** (Sec. III-D): linear probe (LP), MLP probe
      (one 64-unit GELU layer), LoRA ("rank-8 updates with :math:`\alpha` =
      16 in the attention and feed-forward linear layers"), full fine-tuning
      (Full-FT), and the same architecture from random initialization
      (Scratch-FT). Only the linear head is built here.

    .. rubric:: Variants

    The paper pretrains three token durations (Sec. V-B, Table 4; LOSO
    macro-F1, %, each downstream example 10 s, temporal-mean readout); only
    the 10-s weights are released.

    =============  ==================  ============  ===========  =============  ===================
    Token length   ``window_samples``  Gunther 2014  Shader 2021  SujinBak 2019  Weights
    =============  ==================  ============  ===========  =============  ===================
    2 s            20                  78.0          36.6         62.3           not released
    5 s            50                  82.0          43.1         65.0           not released
    10 s           100                 87.3          44.5         67.9           ``mtrenirs_10s.pt``
    =============  ==================  ============  ===========  =============  ===================

    .. rubric:: Pretrained Weights

    ``mtrenirs_10s.pt`` of `microsoft/MtReNIRS
    <https://github.com/microsoft/MtReNIRS>`_ (commit 16c3082, MIT), the
    paper's epoch-95 checkpoint of the 10-s model pretrained on the 12
    datasets of Sec. III-A, is mirrored unchanged at
    `braindecode/mtrenirs-pretrained
    <https://huggingface.co/braindecode/mtrenirs-pretrained>`_. It holds the
    inference encoder only (``{"config", "model"}``; decoder and optimizer
    removed). :meth:`load_state_dict` takes this payload strictly with the
    authors' key names (no ``mapping``); the only new keys,
    ``final_layer.weight`` and ``final_layer.bias``, keep their
    initialization. It expects 10 Hz HbO/HbR, preprocessed as in Notes, in
    100-sample windows, with positions on 10-05 slots:

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

    .. rubric:: Differences from the reference implementation

    - Input is a ``(batch, n_chans, n_times)`` tensor with the montage in
      ``chs_info``, as the reference's ``MtReNIRS.embed`` with ``ch_types``
      and ``ch_pos_1005`` bound in ``from_pretrained``; the reference returns
      features only. ``final_layer`` and ``pool="channel_temporal_mean"``
      (for ``ch_mask``) are added.
    - :math:`q_c` is read from ``chs_info`` as given; the reference's
      ``from_mne_raw`` also snaps it to the nearest 10-05 slot
      (``mtrenirs.alignment.align``) and runs the preprocessing, which stays
      outside this class.
    - A window with a NaN sample is left out of the attention and of the
      ``"temporal_mean"`` average (an addition); channels without a location
      are left out as in the reference.
    - ``on_non_divisible="crop"``/``"pad"`` are the reference's
      ``remainder="drop"``/``"pad"``; its ``"overlap"`` is not offered.
    - Parity: on the released checkpoint, the encoder features equal the
      authors' ``MtReNIRSEncoder`` (MtReNIRS@16c3082) with max-abs difference
      0.0, with all channels located and with NaN or all-zero locations.

    .. versionadded:: 2.1

    `License <https://github.com/microsoft/MtReNIRS/blob/main/LICENSE.txt>`_

    Parameters
    ----------
    window_samples : int, default=100
        Samples :math:`P` per token window :math:`x_{c,t}`; the paper's
        "10-s (100-sample) windows" at 10 Hz (20 and 50 for the 2-s and 5-s
        variants). Authors' config ``data.t_win``.
    embed_dim : int, default=128
        Token and Transformer width :math:`D`, the paper's "128-dimensional
        state" (:math:`z_{c,t} \in \mathbb{R}^{128}`); ``model.d_model``.
    depth : int, default=6
        Number of Transformer blocks, "six pre-layer-normalized Transformer
        blocks"; ``model.encoder.depth``.
    num_heads : int, default=4
        Attention heads :math:`H`, "four attention heads"; ``embed_dim //
        num_heads`` must be even for RoPE. ``model.encoder.n_heads``.
    mlp_dim : int, default=512
        Hidden width of each block's feed-forward network, "a 512-dimensional
        MLP"; ``model.encoder.mlp_hidden``.
    n_fourier_bands : int, default=6
        Fourier bands :math:`2^k \pi`, :math:`k = 0, \dots, 5`, per
        coordinate of :math:`u_c` ("six bands of Fourier features", 48
        features); ``model.fourier_num_bands``.
    spatial_hidden_dim : int, default=128
        Hidden width of the spatio-chromophore MLP, the middle 128 of the
        paper's 48 -> 128 -> 128; ``model.spatial_mlp_hidden``.
    rope_base : float, default=500.0
        RoPE base :math:`\beta` of the window index, "RoPE [...] with base
        500"; ``model.rope_base``.
    qk_norm : bool, default=True
        Per-head LayerNorm of queries and keys, "Per-head layer normalization
        stabilized queries and keys"; ``model.qk_norm``.
    pool : {"temporal_mean", "channel_temporal_mean", "flatten"}, default="temporal_mean"
        Readout of the ``(batch, n_windows, n_chans, embed_dim)`` tokens:
        ``"temporal_mean"``, their mean over the valid windows (the paper's
        temporal averaging); ``"channel_temporal_mean"``, their mean over the
        valid windows and channels (takes ``ch_mask``); ``"flatten"``, all
        tokens flattened (the reference's ``"flatten"``).
    on_non_divisible : {"crop", "pad", "error"}, default="crop"
        A short last window is dropped (the reference's ``remainder="drop"``),
        zero-padded (``"pad"``) or refused.
    coordinate_center : tuple of float
        Centre :math:`\mu` (m) subtracted from :math:`q_c`, the "one center
        [...] computed from the pretraining corpus"; the checkpoint's
        ``data.pos_center``.
    coordinate_scale : float
        RMS scale :math:`\sigma` (m) dividing the centred :math:`q_c`; the
        checkpoint's ``data.pos_scale``.
    drop_prob : float, default=0.1
        Dropout of attention weights, attention output and MLP, the paper's
        "0.1 dropout"; ``model.encoder.dropout``.
    activation : type[nn.Module], default=nn.GELU
        Activation of the spatio-chromophore MLP ("with GELU activation") and
        of the feed-forward blocks.

    Notes
    -----
    Preprocessing (Sec. III-B, Fig. 2A): S-D pairs 8-35 mm apart; optical
    density, scalp-coupling index below 0.5 marked, temporal derivative
    distribution repair, modified Beer-Lambert law with a partial pathlength
    factor of 6.0 (skipped for HbO/HbR releases); "We then band-pass filtered
    each run from 0.01 to 0.5 Hz, resampled to 10 Hz, and removed flat or
    abnormally low-variance chromophore channels"; z-score of each
    chromophore channel over its run, clip to :math:`[-6, 6]`; downstream
    windows "[onset, onset + 10 s) without a hemodynamic-response shift".

    Pretraining (Sec. III-A, III-C): 12 datasets, 268 subject entries, 1,389
    runs, over 567 h, 203,420 non-overlapping 10-s windows, subject-disjoint
    80/20 split; each sample a uniformly drawn dataset and run, a 60-s crop of
    six windows, at most 256 chromophore channels (1,536 tokens); 25 % of
    tokens masked; AdamW, 100 epochs, batch 128, weight decay 0.01, peak
    learning rate :math:`2 \times 10^{-4}`, 10 % linear warm-up then cosine
    decay; the epoch-95 checkpoint has the lowest validation loss.

    Evaluation (Sec. III-D, III-E) on three held-out datasets, among them
    SujinBak 2019 :footcite:`bak2019ufft` (30 subjects, left-hand, right-hand
    and foot tapping, 2,250 10-s samples). Benchmark 1, LOSO: about 15 % of
    the other subjects for inner validation, 50 epochs, AdamW, learning rate
    :math:`10^{-4}` with cosine annealing, weight decay :math:`10^{-4}`,
    batch 32, the state with the best validation macro-F1 tested once.
    Benchmark 2, within-subject few-shot: :math:`N \in \{2, 4, 6, 8\}`
    examples per class, 50 draws, tested on the subject's other examples; the
    linear probe gets 200 full-batch updates on cached frozen features.
    Metric: macro-F1, mean and population SD over subjects.

    The band-pass must remove content above 0.5 Hz. The released
    ``from_mne_raw`` calls MNE's default FIR ``filter(0.01, 0.5)``, which
    passes 0.87 of the amplitude at 1 Hz and leaves 43.8 % of the window
    power above 0.5 Hz (1.1 % with a zero-phase Butterworth); with it the
    LOSO linear probe gives 61.1 and the 2-shot probe 35.1 instead of 68.3
    and 40.3 (:gh:`1564`). A zero-phase Butterworth reproduces the paper:
    ``raw.filter(0.01, 0.5, method="iir", iir_params=dict(order=3, ftype="butter"))``.

    Replication on SujinBak 2019 with this port and ``mtrenirs_10s.pt``
    (macro-F1, %), under the paper's protocol: 4 of the 29 training subjects
    for inner validation; few-shot head learning rate :math:`10^{-4}`; LP and
    MLP 3 seeds (LP 68.8 / 67.4 / 68.8, MLP 67.7 / 67.0 / 67.4), the others 1;
    channels on their nearest 10-05 slots around C3/C4. The authors' encoder
    gives the port's features exactly (max-abs difference 0.0) on these
    windows.

    =========  =====  ====  ====  =======  ==========  ======  ======  ======  ======
    \          LP     MLP   LoRA  Full-FT  Scratch-FT  2-shot  4-shot  6-shot  8-shot
    =========  =====  ====  ====  =======  ==========  ======  ======  ======  ======
    Paper      67.9   67.2  70.3  70.2     66.7        40.4    46.2    50.0    53.0
    Paper SD   17.7   18.6  18.6  19.1     16.2        8.1     11.9    14.2    15.1
    Port       68.3   67.4  69.3  69.6     67.8        40.3    46.6    50.4    53.0
    Rel. diff  +0.6   +0.2  -1.5  -0.8     +1.6        -0.2    +0.9    +0.9    -0.0
    =========  =====  ====  ====  =======  ==========  ======  ======  ======  ======

    The relative difference is in % of the paper value.

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
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_dim: int = 512,
        n_fourier_bands: int = 6,
        spatial_hidden_dim: int = 128,
        rope_base: float = 500.0,
        qk_norm: bool = True,
        pool: Literal[
            "temporal_mean", "channel_temporal_mean", "flatten"
        ] = "temporal_mean",
        on_non_divisible: Literal["crop", "pad", "error"] = "crop",
        coordinate_center: tuple[float, float, float] = (
            0.00021245056996122003,
            -0.022675372660160065,
            0.020025914534926414,
        ),
        coordinate_scale: float = 0.09265176206827164,
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
        if embed_dim % num_heads or (embed_dim // num_heads) % 2:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads}) with an even head_dim for rotary position "
                "embeddings."
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
            window_samples, embed_dim, n_fourier_bands, spatial_hidden_dim, activation
        )
        drop_prob = float(drop_prob)
        blocks = [
            _TransformerBlock(
                embed_dim, num_heads, mlp_dim, drop_prob, rope_base, qk_norm, activation
            )
            for _ in range(depth)
        ]
        self.encoder = _Encoder(blocks, embed_dim)
        default_coordinates = _coordinates(
            self._chs_info, coordinate_center, coordinate_scale
        )
        self.register_buffer(
            "default_coordinates",
            torch.from_numpy(default_coordinates),
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
            "temporal_mean": self.n_chans * embed_dim,
            "channel_temporal_mean": embed_dim,
            "flatten": self.n_windows * self.n_chans * embed_dim,
        }[pool]
        self.final_layer = nn.Linear(self.feature_dim, self.n_outputs)

    def encode_windows(
        self,
        x: torch.Tensor,
        coordinates: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(batch, n_chans, n_times)`` -> tokens ``(batch, n_windows, n_chans,
        embed_dim)`` and their validity ``(batch, n_windows, n_chans)``.
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
        # -> (batch, n_windows * n_chans, embed_dim): z_{c,t}
        embedded = self.embed(tokens, coord_tokens)
        encoded = self.encoder(embedded, window_indices, valid_tokens)
        token_weight = self.token_over_features(valid_tokens)
        token_weight = token_weight.to(encoded.dtype)
        encoded = encoded * token_weight
        # (batch, n_windows * n_chans, embed_dim)
        # -> (batch, n_windows, n_chans, embed_dim)
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
        # (batch, n_chans, n_times) -> (batch, n_windows, n_chans, embed_dim)
        encoded, valid = self.encode_windows(x, coordinates, ch_mask)
        if self.pool == "flatten":
            # -> (batch, n_windows * n_chans * embed_dim)
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
            # -> (batch, n_chans, embed_dim) or (batch, embed_dim)
            features = total / count
            if self.pool == "temporal_mean":
                # (batch, n_chans, embed_dim) -> (batch, n_chans * embed_dim)
                features = self.flatten_channels(features)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)


class _TokenEmbedding(nn.Module):
    """Signal embedding of the window plus the spatio-chromophore embedding
    (an MLP of the Fourier features of :math:`u_c`)."""

    def __init__(
        self, window_samples, embed_dim, n_fourier_bands, spatial_hidden_dim, activation
    ):
        super().__init__()
        # ``signal.proj`` and ``spatial.mlp`` are the checkpoint's key paths
        self.signal = nn.Sequential(
            OrderedDict(proj=nn.Linear(window_samples, embed_dim))
        )
        # 2^k pi, k = 0, ..., n_fourier_bands - 1
        freqs = (2.0 ** torch.arange(n_fourier_bands).float()) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)
        mlp = nn.Sequential(
            nn.Linear(4 * n_fourier_bands * 2, spatial_hidden_dim),
            activation(),
            nn.Linear(spatial_hidden_dim, embed_dim),
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
        # -> (batch, n_tokens, 4, n_fourier_bands): 2^k pi u_{c,j}
        angles = coordinates * freqs
        # -> (batch, n_tokens, 4 * n_fourier_bands)
        angles = self.merge_bands(angles)
        # (batch, n_tokens, window_samples) -> (batch, n_tokens, embed_dim)
        signal_embedding = self.signal(signal)
        sine = angles.sin()
        cosine = angles.cos()
        # -> (batch, n_tokens, 2 * 4 * n_fourier_bands): phi(u_c)
        fourier = torch.cat([sine, cosine], dim=-1)
        # -> (batch, n_tokens, embed_dim)
        spatial_embedding = self.spatial(fourier)
        return signal_embedding + spatial_embedding


class _RotarySelfAttention(TimmAttention):
    """The authors' ``MultiHeadAttention``: timm attention with per-head QK
    LayerNorm, a rotary embedding of the window index and a key padding mask.

    Private because :class:`~braindecode.modules.TimmAttention`'s forward takes
    neither per-token positions nor a mask; its ``qkv``/``proj`` are reused.
    """

    def __init__(self, embed_dim, num_heads, drop_prob, rope_base, qk_norm):
        super().__init__(embed_dim, num_heads, qkv_bias=True, attn_drop=drop_prob)
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
        # (batch, n_tokens, embed_dim) -> 3 x (batch, heads, n_tokens, head_dim)
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
        # (batch, heads, n_tokens, head_dim) -> (batch, n_tokens, embed_dim)
        output = self.merge_heads(output)
        return self.proj(output)


class _TransformerBlock(nn.Module):
    """Pre-norm block: QK-norm rotary self-attention, then feed-forward.

    Private because its attention takes per-token positions and a validity mask
    (:class:`~braindecode.modules.PreNormBlock` calls ``attn(x)`` only) and the
    released ``mtrenirs_10s.pt`` keys follow this layout.
    """

    def __init__(
        self, embed_dim, num_heads, mlp_dim, drop_prob, rope_base, qk_norm, activation
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _RotarySelfAttention(
            embed_dim, num_heads, drop_prob, rope_base, qk_norm
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForwardBlock(
            embed_dim, 1, drop_prob, activation, hidden_features=mlp_dim
        )
        self.drop = nn.Dropout(drop_prob)

    def forward(
        self, values: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        # (batch, n_tokens, embed_dim) -> (batch, n_tokens, embed_dim)
        normed = self.norm1(values)
        attended = self.attn(normed, positions, valid)
        attended = self.drop(attended)
        values = values + attended
        normed = self.norm2(values)
        update = self.mlp(normed)
        return values + update


class _Encoder(nn.Module):
    """The Transformer blocks and a final LayerNorm."""

    def __init__(self, blocks: list[nn.Module], embed_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, values: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        # (batch, n_tokens, embed_dim) -> (batch, n_tokens, embed_dim)
        for block in self.blocks:
            values = block(values, positions, valid)
        return self.norm(values)


def _coordinates(chs_info, coordinate_center, coordinate_scale) -> np.ndarray:
    """``(n_chans, 4)`` :math:`u_c`: ``loc[:3]`` minus ``coordinate_center``,
    divided by ``coordinate_scale`` (NaN when unknown or all zero), and 0 for
    HbO, 1 for HbR, as the official ``chromophore_map``; ``(0, 4)`` without
    ``chs_info`` or any location."""
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
    center = np.asarray(coordinate_center, dtype=np.float32)
    xyz = (loc - center) / float(coordinate_scale)
    is_hbr = np.equal(chromophores, "hbr")
    return np.c_[xyz, is_hbr].astype(np.float32)
