from unittest.mock import Mock

import torch

from braindecode.modules.parametrization import MaxNormParametrize


def test_max_norm_parametrize_avoids_unsupported_renorm(monkeypatch) -> None:
    weights = [
        torch.tensor([[3.0, 4.0], [0.0, 0.0]]),
        torch.randn(3, 2, 4, 5),
    ]
    expected = [value.renorm(p=2, dim=0, maxnorm=1.0) for value in weights]
    blocked_renorm = Mock(side_effect=AssertionError("unsupported renorm"))
    monkeypatch.setattr(torch.Tensor, "renorm", blocked_renorm)

    actual = [MaxNormParametrize(max_norm=1.0)(value) for value in weights]

    blocked_renorm.assert_not_called()
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(actual_value, expected_value)
