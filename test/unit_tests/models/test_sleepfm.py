# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

"""Tests for the released SleepFM embedding-input sleep stager."""

from __future__ import annotations

import importlib.util
import os
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from torch import nn

from braindecode.models import SleepFMStager

_OFFICIAL_COMMIT = "2bcbae04c3592f61352addb7ac3d4193f0a3ca25"
_STAGING_MANIFEST_SHA256 = (
    "f09ea88810b1c4e0201aabf27772c40378c35bf54bd52ca8f1df6fa38d9d40ce"
)


def _official_root():
    configured = os.environ.get("BRAINDECODE_SLEEPFM_OFFICIAL_SOURCE")
    if configured:
        return Path(configured)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "experiments/model-replication-sources/SleepFM-official"
        if candidate.exists():
            return candidate
    return Path("__sleepfm_official_source_not_found__")


def _small_stager(**kwargs):
    defaults = dict(
        n_outputs=5,
        n_chans=3,
        n_times=6,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        pooling_heads=4,
        drop_prob=0.0,
        max_seq_length=16,
    )
    return SleepFMStager(**(defaults | kwargs))


def _source_staging_state(model):
    state = OrderedDict()
    for key, value in model.state_dict().items():
        if key.startswith("final_layer."):
            key = key.replace("final_layer.", "fc.", 1)
        state[f"module.{key}"] = value.detach().clone()
    return state


def _manifest_digest(state):
    manifest = "\n".join(
        f"{key}:{state[key].dtype}:{','.join(map(str, state[key].shape))}"
        for key in sorted(state)
    )
    return sha256(manifest.encode()).hexdigest()


def test_sleepfm_stager_accepts_official_embedding_and_mask_shapes():
    model = _small_stager().eval()
    embeddings = torch.randn(2, 3, 6, 16)
    padding_mask = torch.zeros(2, 3, 6)
    padding_mask[0, 2] = 1
    padding_mask[1, :, 4:] = 1

    with torch.no_grad():
        logits = model(embeddings, padding_mask)

    assert logits.shape == (2, 6, 5)


def test_sleepfm_stager_rejects_raw_signal_input():
    model = _small_stager().eval()

    with pytest.raises(ValueError, match="4D precomputed embeddings"):
        model(torch.randn(2, 3, 640))


@pytest.mark.parametrize(
    "embeddings,padding_mask,message",
    [
        (torch.randn(2, 3, 6, 8), None, "embedding axis"),
        (torch.randn(2, 3, 17, 16), None, "max_seq_length"),
        (torch.randn(2, 0, 6, 16), None, "at least one modality"),
        (torch.randn(2, 3, 6, 16), torch.zeros(2, 3, 5), "same first three"),
        (
            torch.randn(2, 3, 6, 16),
            torch.zeros(2, 3, 6, dtype=torch.int64),
            "boolean or floating point",
        ),
    ],
)
def test_sleepfm_stager_validates_embedding_boundary(embeddings, padding_mask, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _small_stager().eval()(embeddings, padding_mask)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"embed_dim": 15}, "even"),
        ({"embed_dim": 14, "num_heads": 4}, "divisible"),
        ({"embed_dim": 14, "pooling_heads": 4}, "divisible"),
        ({"num_heads": 0}, "positive integer"),
        ({"num_layers": 0}, "positive integer"),
        ({"pooling_heads": True}, "positive integer"),
        ({"drop_prob": -0.1}, r"\[0, 1\]"),
        ({"activation": nn.ReLU()}, "nn.Module class"),
    ],
)
def test_sleepfm_stager_validates_architecture(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _small_stager(**kwargs)


def test_sleepfm_staging_manifest_matches_pinned_release():
    model = SleepFMStager(n_outputs=5)
    source_state = _source_staging_state(model)

    assert len(source_state) == 37
    assert _manifest_digest(source_state) == _STAGING_MANIFEST_SHA256
    model.load_released_weights(source_state)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "shape", "dtype"])
def test_sleepfm_loader_rejects_nonexact_staging_manifest(mutation):
    model = SleepFMStager(n_outputs=5)
    source_state = _source_staging_state(model)
    if mutation == "missing":
        source_state.pop("module.fc.bias")
    elif mutation == "unknown":
        source_state["module.unknown"] = torch.zeros(1)
    elif mutation == "shape":
        source_state["module.fc.bias"] = torch.zeros(6)
    else:
        source_state["module.fc.bias"] = source_state["module.fc.bias"].double()

    with pytest.raises(RuntimeError, match="37-tensor staging checkpoint manifest"):
        model.load_released_weights(source_state)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_outputs": 4},
        {"embed_dim": 64},
        {"num_heads": 2},
        {"num_layers": 2},
        {"pooling_heads": 2},
        {"drop_prob": 0.0},
        {"max_seq_length": 128},
        {"activation": nn.GELU},
    ],
)
def test_sleepfm_loader_rejects_nonreleased_graph(kwargs):
    model = SleepFMStager(**({"n_outputs": 5} | kwargs))

    with pytest.raises(ValueError, match="official released configuration"):
        model.load_released_weights({})


def test_sleepfm_reset_head_preserves_dtype_and_config():
    model = _small_stager().double()

    model.reset_head(3)

    assert model.n_outputs == 3
    assert model.final_layer.out_features == 3
    assert model.final_layer.weight.dtype == torch.float64
    assert model.get_config()["n_outputs"] == 3
    assert SleepFMStager.from_config(model.get_config()).n_outputs == 3


def _load_official_model_class():
    official_root = _official_root()
    source_file = official_root / "sleepfm/models/models.py"
    checkpoint = official_root / "sleepfm/checkpoints/model_sleep_staging/best.pth"
    if not source_file.exists() or not checkpoint.exists():
        pytest.skip(
            "Set BRAINDECODE_SLEEPFM_OFFICIAL_SOURCE to the pinned official clone."
        )
    head = (official_root / ".git/HEAD").read_text().strip()
    assert head == _OFFICIAL_COMMIT
    spec = importlib.util.spec_from_file_location(
        "_sleepfm_official_models", source_file
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SleepEventLSTMClassifier, checkpoint


@pytest.mark.parametrize("padded", [False, True])
def test_sleepfm_matches_pinned_official_source(padded):
    official_class, checkpoint_path = _load_official_model_class()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    official = official_class(
        embed_dim=128,
        num_heads=4,
        num_layers=1,
        num_classes=5,
        pooling_head=4,
        dropout=0.3,
        max_seq_length=8196,
    ).eval()
    official.load_state_dict(
        {key.removeprefix("module."): value for key, value in checkpoint.items()},
        strict=True,
    )
    model = SleepFMStager(n_outputs=5).eval()
    model.load_released_weights(checkpoint)

    generator = torch.Generator().manual_seed(20260825)
    embeddings = torch.randn(2, 3, 6, 128, generator=generator)
    padding_mask = torch.zeros(2, 3, 6)
    if padded:
        padding_mask[0, 2] = 1
        padding_mask[1, :, 4:] = 1
        embeddings = embeddings.masked_fill(padding_mask.bool().unsqueeze(-1), 0)

    with torch.no_grad():
        expected, expected_mask = official(embeddings, padding_mask)
        actual = model(embeddings, padding_mask)

    assert torch.equal(expected_mask, padding_mask[:, 0])
    assert torch.equal(actual, expected)
