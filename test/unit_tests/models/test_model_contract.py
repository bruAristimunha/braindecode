"""Cross-model contract gates for every registered Braindecode model.

These tests turn the repository model conventions into machine-enforced
invariants. A newly registered model is automatically included through
models_mandatory_parameters; no per-model test opt-in is required.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import pickle
import re
from collections.abc import Mapping, Sequence

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from braindecode.models.util import (
    _get_signal_params,
    models_dict,
    models_mandatory_parameters,
)

all_models_dict = dict(models_dict)


def _tensor_leaves(value):
    """Yield tensor leaves from nested model outputs."""
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _tensor_leaves(child)


def _build_case(model_name, required_params, signal_params):
    signal = _get_signal_params(signal_params)
    model_kwargs = _get_signal_params(signal_params, required_params)
    model = all_models_dict[model_name](**model_kwargs).eval()
    x = torch.randn(2, len(signal["chs_info"]), signal["n_times"])
    return model, x


def _materialize(model, x):
    """Run one no-grad forward if lazy parameters still need their shapes."""
    if any(isinstance(p, nn.UninitializedParameter) for p in model.parameters()):
        with torch.no_grad():
            model(x)


def _clone_state(model):
    """Clone persistent tensor state after any lazy first-forward setup."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _batched_tensor_leaves(value, batch_size):
    """Return tensor leaves whose leading dimension represents the batch."""
    return [
        leaf
        for leaf in _tensor_leaves(value)
        if leaf.ndim > 0 and leaf.shape[0] == batch_size
    ]


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    models_mandatory_parameters,
)
def test_registered_model_runtime_contract(
    model_name, required_params, signal_params
):
    """Registered models preserve inputs and emit finite batched tensors."""
    model, x = _build_case(model_name, required_params, signal_params)
    x_before = x.clone()

    with torch.no_grad():
        output = model(x)

    assert torch.equal(x, x_before), (
        f"{model_name} mutated its input tensor during eval-mode forward"
    )

    leaves = list(_tensor_leaves(output))
    assert leaves, f"{model_name} returned no tensor output"

    batched = _batched_tensor_leaves(output, x.shape[0])
    assert batched, (
        f"{model_name} returned no tensor leaf preserving batch dimension "
        f"{x.shape[0]}"
    )

    for leaf in leaves:
        if leaf.is_floating_point() or leaf.is_complex():
            assert torch.isfinite(leaf).all(), (
                f"{model_name} emitted non-finite values in eval-mode forward"
            )

    # A first forward may legitimately materialize lazy parameters. Once warm,
    # however, eval-mode inference must not mutate persistent model state.
    # Reuse the permutation probe below as the second forward so the gate stays
    # cheap even for large foundation models.
    state_before = _clone_state(model)

    # Reordering independent samples must only reorder the corresponding
    # outputs. This catches accidental batch-axis mixing that ordinary shape
    # checks and single-sample tests cannot see.
    permutation = torch.tensor([1, 0], device=x.device)
    with torch.no_grad():
        permuted = model(x.index_select(0, permutation))

    state_after = model.state_dict()
    assert state_before.keys() == state_after.keys()
    for name, expected_state in state_before.items():
        torch.testing.assert_close(
            state_after[name],
            expected_state,
            msg=lambda msg: (
                f"{model_name} mutated persistent state {name!r} during "
                f"eval-mode forward: {msg}"
            ),
        )

    permuted_batched = _batched_tensor_leaves(permuted, x.shape[0])
    assert len(permuted_batched) == len(batched)
    for expected_leaf, actual_leaf in zip(batched, permuted_batched):
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf.index_select(0, permutation),
            msg=lambda msg: (
                f"{model_name} is not batch-permutation equivariant in eval "
                f"mode: {msg}"
            ),
        )
    # Permutation equivariance alone cannot detect all cross-sample mixing: a
    # symmetric batch aggregate can influence every output and still permute
    # correctly. Keep sample 0 fixed, change only sample 1, and require sample
    # 0's outputs to remain invariant.
    composed_x = x.clone()
    composed_x[1].mul_(-3.0).add_(1.0)
    with torch.no_grad():
        recomposed = model(composed_x)

    recomposed_batched = _batched_tensor_leaves(recomposed, x.shape[0])
    assert len(recomposed_batched) == len(batched)
    for expected_leaf, actual_leaf in zip(batched, recomposed_batched):
        torch.testing.assert_close(
            actual_leaf[0],
            expected_leaf[0],
            msg=lambda msg: (
                f"{model_name} leaked information across independent batch "
                f"samples in eval mode: {msg}"
            ),
        )


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    models_mandatory_parameters,
)
def test_registered_model_serialization_contract(
    model_name, required_params, signal_params
):
    """Config + state_dict reconstruction, deepcopy and pickle preserve outputs."""
    model, x = _build_case(model_name, required_params, signal_params)

    with torch.no_grad():
        expected = list(_tensor_leaves(model(x)))

    config = model.get_config()
    serialized = json.dumps(config)
    rebuilt = type(model).from_config(json.loads(serialized)).eval()
    _materialize(rebuilt, x)
    rebuilt.load_state_dict(model.state_dict(), strict=True)

    copies = {"config/state round-trip": rebuilt, "deepcopy": copy.deepcopy(model)}
    # torch refuses to pickle modules with parametrizations (weight_norm,
    # max-norm constraints): those models are saved through state_dict only.
    if not any(parametrize.is_parametrized(m) for m in model.modules()):
        copies["pickle"] = pickle.loads(pickle.dumps(model))
    for how, clone in copies.items():
        with torch.no_grad():
            actual = list(_tensor_leaves(clone(x)))
        assert len(actual) == len(expected), (
            f"{model_name} changed tensor-output structure after {how}"
        )
        for expected_leaf, actual_leaf in zip(expected, actual):
            assert expected_leaf.shape == actual_leaf.shape
            torch.testing.assert_close(actual_leaf, expected_leaf)


# Trainable parameters the default forward does not reach (pretraining heads,
# other read-outs, reference quirks); kept so released checkpoints load.
_UNUSED_IN_FORWARD = {
    "AttnSleep": r"self_attn\.convs\.0\.",  # the reference never convolves the query
    "BrainOmni": r"^blocks\.11\.",  # the reference encode() skips the last block
    "Brant": r"spatial_encoder\.proj_out\.",  # reconstruction head
    "CodeBrain": r"residual_blocks\.7\.(rms_norm|res_conv)|^lm_head_",  # skip-only last block, tokenizer heads
    "DANCE": r"^decoder\.",  # event decoder of detect()
    "MSVTNet": r"^branch_head\.",  # auxiliary branch heads (return_features)
    "PopulationTransformer": r"^spec_prediction_head\.",  # pretraining head
    "SignalJEPA": r"^transformer\.decoder\.",  # pretraining decoder
    "SignalJEPA_Contextual": r"^transformer\.decoder\.",
    "SSTDPN": r"^proto_cpt$",  # prototype loss term
    "STEEGFormer": r"^norm\.",  # only the cls read-out normalises
}


def _train_step(model, x, sync=lambda: None, unused=None):
    """One train-mode SGD step; every trainable parameter gets a finite gradient."""
    model.train()
    leaves = [t for t in _tensor_leaves(model(x)) if t.requires_grad]
    assert leaves, "no differentiable output"
    assert all(t.shape[0] == x.shape[0] for t in leaves if t.ndim)
    loss = sum(t.float().square().mean() for t in leaves)
    assert torch.isfinite(loss), "non-finite train-mode output"
    loss.backward()
    sync()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    no_grad = [
        n
        for n, p in trainable.items()
        if p.grad is None and not (unused and re.search(unused, n))
    ]
    assert not no_grad, f"no gradient reaches {no_grad}"
    bad = [
        n
        for n, p in trainable.items()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not bad, f"non-finite gradient in {bad}"
    torch.optim.SGD(trainable.values(), lr=1e-3).step()
    sync()
    assert all(torch.isfinite(p).all() for p in trainable.values())


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    models_mandatory_parameters,
)
def test_registered_model_training_contract(
    model_name, required_params, signal_params
):
    """A model used under ``torch.inference_mode`` still trains, then evaluates."""
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    with torch.inference_mode():
        expected = _batched_tensor_leaves(model(x), x.shape[0])

    _train_step(model, x, unused=_UNUSED_IN_FORWARD.get(model_name))

    # e.g. keeping the best model: no non-leaf tensor may stay cached.
    model = copy.deepcopy(model)
    with torch.no_grad():
        actual = _batched_tensor_leaves(model.eval()(x), x.shape[0])
    assert [t.shape for t in actual] == [t.shape for t in expected]
    assert all(torch.isfinite(t).all() for t in actual if t.is_floating_point())


_DATA_DEPENDENT_FORWARD = {"BaRISTA", "BrainOmni", "BrainTokenizer", "CodeBrain"}


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    models_mandatory_parameters,
)
def test_registered_model_follows_device(model_name, required_params, signal_params):
    """``model.to(device)`` moves every tensor forward reads (checked on ``meta``)."""
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    model.to("meta")
    stray = [
        f"{prefix}.{attr}".lstrip(".")
        for prefix, module in model.named_modules()
        for attr, value in vars(module).items()
        if torch.is_tensor(value) and not value.is_meta
    ]
    assert not stray, f"tensor attributes that .to() does not move: {stray}"
    if model_name in _DATA_DEPENDENT_FORWARD:
        pytest.skip("forward reads tensor values (.item()), which meta tensors lack")
    with torch.no_grad():
        output = model(x.to("meta"))
    assert all(t.is_meta for t in _tensor_leaves(output))


# Gaudi (habana_frameworks); run with ``pytest -m hpu`` on a Gaudi host, once
# with PT_HPU_LAZY_MODE=1 (lazy, the default) and once with 0 (eager).
_HPU_MODE = "eager" if os.environ.get("PT_HPU_LAZY_MODE") == "0" else "lazy"
# Gaudi software 1.21 (Synapse) failures with no portable equivalent op; the
# same models run on CPU. (mode, reason); "slow" cells are not run.
_HPU_KNOWN = {
    "BrainModule": ("lazy", "graph compile fails in forward (synStatus 26)"),
    "VEMG2Pose": ("lazy", "graph compile fails in forward (synStatus 26)"),
    "CodeBrain": ("lazy", "graph compile fails in backward (synStatus 26)"),
    "DANCE": ("lazy", "graph compile fails in backward (synStatus 26)"),
    "SensingDynamics": ("lazy", "bfloat16 output is not finite"),
    "EEGSym": ("lazy", "slow: graph compile takes > 300 s"),
    "SignalJEPA_PreLocal": ("lazy", "slow: graph compile takes > 300 s"),
    "MEDFormer": ("both", "slow: > 300 s (gaudi2_agu_config: size - 1 <= uint8 max)"),
    "USleep": ("eager", "graph compile fails at a length-1 Upsample"),
}


def _hpu_cases():
    for name, required, signal in models_mandatory_parameters:
        mode, reason = _HPU_KNOWN.get(name, (None, ""))
        marks = []
        if mode in ("both", _HPU_MODE):
            marks = pytest.mark.xfail(reason=reason, run=not reason.startswith("slow"))
        yield pytest.param(name, required, signal, marks=marks, id=name)


@pytest.mark.hpu
@pytest.mark.skipif(
    importlib.util.find_spec("habana_frameworks") is None, reason="needs a Gaudi HPU"
)
@pytest.mark.parametrize("model_name,required_params,signal_params", list(_hpu_cases()))
def test_registered_model_on_hpu(model_name, required_params, signal_params):
    """Forward and one train step on HPU in float32 and bfloat16."""
    import habana_frameworks.torch.core as htcore

    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    for dtype in (torch.float32, torch.bfloat16):
        hpu_model = copy.deepcopy(model).to("hpu", dtype).eval()
        hpu_x = x.to("hpu", dtype)
        with torch.no_grad():
            leaves = list(_tensor_leaves(hpu_model(hpu_x)))
        htcore.mark_step()
        assert all(t.device.type == "hpu" for t in leaves)
        assert all(torch.isfinite(t).all() for t in leaves if t.is_floating_point())
        _train_step(
            hpu_model, hpu_x, htcore.mark_step, _UNUSED_IN_FORWARD.get(model_name)
        )
