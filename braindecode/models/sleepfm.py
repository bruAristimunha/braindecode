# Authors: Fashad Ahmed <Fashad-Ahmed@users.noreply.github.com>
#
# Code adapted from https://github.com/zou-group/sleepfm-clinical
#
# License: Creative Commons Attribution-NonCommercial 4.0 International
# This derivative is not covered by Braindecode's BSD-3 license.

"""SleepFM embedding-input sleep-stage inference model."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
from numbers import Real

import torch
from torch import nn

from braindecode.models.base import EEGModuleMixin

_RELEASED_STAGING_TENSOR_COUNT = 37
_RELEASED_STAGING_MANIFEST_SHA256 = (
    "f09ea88810b1c4e0201aabf27772c40378c35bf54bd52ca8f1df6fa38d9d40ce"
)


class SleepFMStager(EEGModuleMixin, nn.Module):
    r"""SleepFM sleep stager from Thapa et al. (2026) [sleepfm2026]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-secondary:`Recurrent`

    This is the released sleep-staging head, not the upstream raw-signal base
    encoder. It accepts precomputed SleepFM embeddings with shape
    ``(batch, modalities, sequence, embed_dim)``. Raw PSG signals must first be
    transformed by the separately licensed upstream base model.

    The official padding mask has shape ``(batch, modalities, sequence)`` and
    uses zero for observed embeddings and one for padding. Boolean and floating
    masks are accepted. As in the released implementation, the spatial mask is
    booleanized, the temporal slice is supplied to the Transformer, and the
    ordinary bidirectional LSTM processes the complete sequence without
    packing or zeroing padded logits.

    The released checkpoint can be loaded locally with
    :meth:`load_released_weights`. The loader accepts only the exact 37-tensor
    staging checkpoint schema and does not download or re-host weights.

    The upstream implementation and weights are licensed under
    `CC BY-NC 4.0 <https://creativecommons.org/licenses/by-nc/4.0/>`_. This
    derivative is not covered by Braindecode's BSD-3 license.

    .. rubric:: Architecture Overview

    1. Self-attention pools the available modality embeddings at each token.
    2. Positional encoding and a Transformer contextualize the token sequence.
    3. A bidirectional LSTM and linear head emit one sleep-stage logit vector
       per token.

    .. rubric:: Macro Components

    ``SleepFMStager.spatial_pooling``
        **Operations:** Transformer encoder layer followed by a masked mean.
        **Role:** combines a variable set of modality embeddings.

    ``SleepFMStager.transformer_encoder``
        **Operations:** pre-normalized self-attention and feed-forward layer.
        **Role:** contextualizes the embedding sequence.

    ``SleepFMStager.lstm`` and ``SleepFMStager.final_layer``
        **Operations:** bidirectional recurrence and a linear projection.
        **Role:** predict a sleep stage at every sequence position.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** positional encoding, self-attention, and bidirectional LSTM.
    - **Channels/space:** masked attention pooling across modalities.
    - **Spectral:** supplied by the external SleepFM base embeddings; this
      staging head performs no raw-signal or spectral preprocessing.

    .. rubric:: Additional Mechanisms

    The mask convention and ordinary, unpacked BiLSTM behavior intentionally
    match official commit ``2bcbae04``.

    .. versionadded:: 1.8.0

    Parameters
    ----------
    embed_dim : int, default=128
        Width of each precomputed embedding. It must be even because the two
        LSTM directions each use ``embed_dim // 2`` hidden units.
    num_heads : int, default=4
        Attention heads in the sequence Transformer.
    num_layers : int, default=1
        Transformer and bidirectional-LSTM layers.
    pooling_heads : int, default=4
        Attention heads used to pool modalities.
    drop_prob : float, default=0.3
        Dropout probability in the attention and recurrent blocks.
    max_seq_length : int, default=8196
        Maximum number of embedding tokens.
    activation : type[nn.Module], default=nn.ReLU
        Transformer feed-forward activation.

    References
    ----------
    .. [sleepfm2026] Thapa, R., Kjaer, M. R., He, B., et al. (2026).
       A multimodal sleep foundation model for disease prediction.
       *Nature Medicine*, 32, 752--762.
       https://doi.org/10.1038/s41591-025-04133-4
    """

    def __init__(
        self,
        # Braindecode parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # Model-specific parameters
        *,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        pooling_heads: int = 4,
        drop_prob: float = 0.3,
        max_seq_length: int = 8196,
        activation: type[nn.Module] = nn.ReLU,
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

        for name, value in (
            ("embed_dim", embed_dim),
            ("num_heads", num_heads),
            ("num_layers", num_layers),
            ("pooling_heads", pooling_heads),
            ("max_seq_length", max_seq_length),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if embed_dim % 2:
            raise ValueError("embed_dim must be even for the bidirectional LSTM.")
        for name, heads in (("num_heads", num_heads), ("pooling_heads", pooling_heads)):
            if embed_dim % heads:
                raise ValueError(f"embed_dim must be divisible by {name}.")
        if (
            not isinstance(drop_prob, Real)
            or isinstance(drop_prob, bool)
            or not 0 <= drop_prob <= 1
        ):
            raise ValueError("drop_prob must be a real number in [0, 1].")
        if not isinstance(activation, type) or not issubclass(activation, nn.Module):
            raise TypeError("activation must be an nn.Module class.")

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.num_layers: int = num_layers
        self.pooling_heads: int = pooling_heads
        self.drop_prob: float = float(drop_prob)
        self.max_seq_length: int = max_seq_length
        self.activation: type[nn.Module] = activation

        self.spatial_pooling: _SleepFMAttentionPooling = _SleepFMAttentionPooling(
            embed_dim,
            num_heads=pooling_heads,
            drop_prob=self.drop_prob,
            activation=activation,
        )
        self.positional_encoding: _SleepFMPositionalEncoding = (
            _SleepFMPositionalEncoding(
                max_seq_length,
                embed_dim,
            )
        )
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=self.drop_prob,
            activation=activation(),
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.drop_prob if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.final_layer: nn.Linear = nn.Linear(embed_dim, self.n_outputs)

    def forward(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return token-wise logits from precomputed SleepFM embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            Tensor shaped ``(batch, modalities, sequence, embed_dim)``.
        padding_mask : torch.Tensor | None
            Official mask shaped ``(batch, modalities, sequence)``. Zero or
            ``False`` is observed; one or ``True`` is padding. ``None`` means
            every embedding is observed.
        """
        if embeddings.ndim != 4:
            raise ValueError(
                "SleepFMStager expects 4D precomputed embeddings with shape "
                "(batch, modalities, sequence, embed_dim)."
            )
        batch, modalities, sequence, embedding_width = embeddings.shape
        if modalities == 0:
            raise ValueError("SleepFMStager requires at least one modality.")
        if embedding_width != self.embed_dim:
            raise ValueError(
                f"The embedding axis must have width {self.embed_dim}; "
                f"got {embedding_width}."
            )
        if sequence > self.max_seq_length:
            raise ValueError(
                f"Embedding sequence length {sequence} exceeds "
                f"max_seq_length={self.max_seq_length}."
            )
        if padding_mask is None:
            padding_mask = torch.zeros(
                (batch, modalities, sequence),
                dtype=torch.bool,
                device=embeddings.device,
            )
        else:
            if padding_mask.shape != embeddings.shape[:3]:
                raise ValueError(
                    "padding_mask must have the same first three axes as embeddings."
                )
            if (
                padding_mask.dtype != torch.bool
                and not padding_mask.is_floating_point()
            ):
                raise TypeError("padding_mask must be boolean or floating point.")
            padding_mask = padding_mask.to(device=embeddings.device)

        tokens = embeddings.permute(0, 2, 1, 3).reshape(
            batch * sequence,
            modalities,
            self.embed_dim,
        )
        spatial_mask = (
            padding_mask[:, :, 0]
            .unsqueeze(1)
            .expand(batch, sequence, modalities)
            .reshape(batch * sequence, modalities)
        )
        features = self.spatial_pooling(tokens, spatial_mask)
        features = features.reshape(batch, sequence, self.embed_dim)
        features = self.positional_encoding(features)
        features = self.layer_norm(features)
        temporal_mask = padding_mask[:, 0, :]
        features = self.transformer_encoder(
            features,
            src_key_padding_mask=temporal_mask,
        )
        features, _ = self.lstm(features)
        return self.final_layer(features)

    def reset_head(self, n_outputs: int):
        """Replace the token-wise classification layer."""
        if (
            not isinstance(n_outputs, int)
            or isinstance(n_outputs, bool)
            or n_outputs <= 0
        ):
            raise ValueError("n_outputs must be a positive integer.")
        old_head = self.final_layer
        self.final_layer = nn.Linear(old_head.in_features, n_outputs).to(
            device=old_head.weight.device,
            dtype=old_head.weight.dtype,
        )
        self._n_outputs = n_outputs
        init_kwargs = getattr(self, "_braindecode_init_kwargs", None)
        if init_kwargs is not None and "n_outputs" in init_kwargs:
            init_kwargs["n_outputs"] = n_outputs
        hub_config = getattr(self, "_hub_mixin_config", None)
        if hub_config is not None and "n_outputs" in hub_config:
            hub_config["n_outputs"] = n_outputs
        return self

    def load_released_weights(self, state_dict: Mapping[str, object]):
        """Strictly load the official local sleep-staging checkpoint."""
        mismatches = [
            name
            for name, actual, expected in (
                ("n_outputs", self.n_outputs, 5),
                ("embed_dim", self.embed_dim, 128),
                ("num_heads", self.num_heads, 4),
                ("num_layers", self.num_layers, 1),
                ("pooling_heads", self.pooling_heads, 4),
                ("drop_prob", self.drop_prob, 0.3),
                ("max_seq_length", self.max_seq_length, 8196),
                ("activation", self.activation, nn.ReLU),
            )
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "SleepFM weights require the official released configuration; "
                f"mismatched settings: {mismatches}."
            )

        source_state = _unwrap_state_dict(state_dict)
        _require_released_staging_manifest(source_state)
        mapped: OrderedDict[str, torch.Tensor] = OrderedDict()
        for source_key, value in source_state.items():
            key = source_key.removeprefix("module.")
            if key.startswith("fc."):
                key = key.replace("fc.", "final_layer.", 1)
            mapped[key] = value
        return self.load_state_dict(mapped, strict=True)


class _SleepFMAttentionPooling(nn.Module):
    """Official self-attention and masked mean across modalities."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 1,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dropout=drop_prob,
            activation=activation(),
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            if key_padding_mask.shape[1] == 1:
                return x.mean(dim=1)
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(dtype=torch.bool)
            output = self.transformer_layer(
                x,
                src_key_padding_mask=key_padding_mask,
            )
            valid = (~key_padding_mask).float().unsqueeze(-1)
            return (output * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        output = self.transformer_layer(x)
        return output.mean(dim=1)


class _SleepFMPositionalEncoding(nn.Module):
    """Official sinusoidal positional encoding."""

    def __init__(self, max_seq_length: int, embed_dim: int) -> None:
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        encoding = torch.zeros(max_seq_length, embed_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


def _unwrap_state_dict(
    state_dict: Mapping[str, object],
) -> Mapping[str, torch.Tensor]:
    nested = state_dict.get("state_dict")
    if isinstance(nested, Mapping):
        state_dict = nested
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError("SleepFM checkpoints may only contain tensors.")
    return state_dict  # type: ignore[return-value]


def _require_released_staging_manifest(
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    manifest = "\n".join(
        f"{key}:{state_dict[key].dtype}:{','.join(map(str, state_dict[key].shape))}"
        for key in sorted(state_dict)
    )
    digest = sha256(manifest.encode()).hexdigest()
    if (
        len(state_dict) != _RELEASED_STAGING_TENSOR_COUNT
        or digest != _RELEASED_STAGING_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "SleepFM 37-tensor staging checkpoint manifest is incompatible; "
            "missing, unexpected, shape-mismatched, and dtype-mismatched "
            "tensors are rejected."
        )
